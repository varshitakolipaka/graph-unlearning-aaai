import torch
import os
import math
import torch.nn as nn
import torch_geometric.transforms as T
from framework.training_args import parse_args
from framework import utils
from models.deletion import GCNDelete
from models.models import GCN
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import is_undirected, to_undirected, negative_sampling, k_hop_subgraph
from trainers.base import EdgeTrainer
from attacks.mi_attack import MIAttackTrainer
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_test_split_edges_no_neg_adj_mask(data, val_ratio: float = 0.05, test_ratio: float = 0.05, two_hop_degree=None):
    '''Avoid adding neg_adj_mask'''

    num_nodes = data.num_nodes
    row, col = data.edge_index
    edge_attr = data.edge_attr

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    perm = torch.randperm(row.size(0))

    row = row[perm]
    col = col[perm]

    # Train
    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.train_pos_edge_index, data.train_pos_edge_attr = None
    else:
        data.train_pos_edge_index = data.train_pos_edge_index
    
    assert not is_undirected(data.train_pos_edge_index)
    
    # Test
    r, c = row[:n_t], col[:n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    neg_edge_index = negative_sampling(
        edge_index=data.test_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.test_pos_edge_index.shape[1])

    data.test_neg_edge_index = neg_edge_index

    # Valid
    r, c = row[n_t:n_t+n_v], col[n_t:n_t+n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)

    neg_edge_index = negative_sampling(
        edge_index=data.val_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.val_pos_edge_index.shape[1])

    data.val_neg_edge_index = neg_edge_index

    return data

def gen_inout_mask(data):
    _, local_edges, _, mask = k_hop_subgraph(
        data.val_pos_edge_index.flatten().unique(), 
        2, 
        data.train_pos_edge_index, 
        num_nodes=data.num_nodes)
    distant_edges = data.train_pos_edge_index[:, ~mask]
    print('Number of edges. Local: ', local_edges.shape[1], 'Distant:', distant_edges.shape[1])

    in_mask = mask
    out_mask = ~mask

    return {'in': in_mask, 'out': out_mask}

def split_forget_retain(data, df_size, subset='in'):
    if df_size >= 100:     # df_size is number of nodes/edges to be deleted
        df_size = int(df_size)
    else:                       # df_size is the ratio
        df_size = int(df_size / 100 * data.train_pos_edge_index.shape[1])
    print(f'Original size: {data.train_pos_edge_index.shape[1]:,}')
    print(f'Df size: {df_size:,}')
    df_mask_all = gen_inout_mask(data)[subset]
    df_nonzero = df_mask_all.nonzero().squeeze()        # subgraph子图内/外的edge idx序号

    idx = torch.randperm(df_nonzero.shape[0])[:df_size]
    df_global_idx = df_nonzero[idx]

    dr_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    dr_mask[df_global_idx] = False

    df_mask = torch.zeros(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    df_mask[df_global_idx] = True

    # Collect enclosing subgraph of Df for loss computation
    # Edges in S_Df
    _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
        data.train_pos_edge_index[:, df_mask].flatten().unique(), 
        2, 
        data.train_pos_edge_index,
        num_nodes=data.num_nodes)
    data.sdf_mask = two_hop_mask

    # Nodes in S_Df
    _, one_hop_edge, _, one_hop_mask = k_hop_subgraph(
        data.train_pos_edge_index[:, df_mask].flatten().unique(), 
        1, 
        data.train_pos_edge_index,
        num_nodes=data.num_nodes)
    sdf_node_1hop = torch.zeros(data.num_nodes, dtype=torch.bool)
    sdf_node_2hop = torch.zeros(data.num_nodes, dtype=torch.bool)

    sdf_node_1hop[one_hop_edge.flatten().unique()] = True
    sdf_node_2hop[two_hop_edge.flatten().unique()] = True

    assert sdf_node_1hop.sum() == len(one_hop_edge.flatten().unique())
    assert sdf_node_2hop.sum() == len(two_hop_edge.flatten().unique())

    data.sdf_node_1hop_mask = sdf_node_1hop
    data.sdf_node_2hop_mask = sdf_node_2hop

    # assert not is_undirected(data.train_pos_edge_index)
    data = data.to(device)
    df_mask = df_mask.to(device)
    train_pos_edge_index, [df_mask, two_hop_mask] = to_undirected(data.train_pos_edge_index, [df_mask.int(), two_hop_mask.int()])
    # to_undirected return full undirected edges and corresponding mask for given edge_attrs
    two_hop_mask = two_hop_mask.bool()  
    df_mask = df_mask.bool()
    dr_mask = ~df_mask

    data.train_pos_edge_index = train_pos_edge_index
    data.edge_index = train_pos_edge_index
    assert is_undirected(data.train_pos_edge_index)

    data.directed_df_edge_index = data.train_pos_edge_index[:, df_mask]
    data.sdf_mask = two_hop_mask
    data.df_mask = df_mask
    data.dr_mask = dr_mask
    return data

def get_processed_data(d, val_ratio=0.05, test_ratio=0.05):
    data_dir = './data' 
    dataset = Planetoid(os.path.join(data_dir, d), d.split('_')[0], transform=T.NormalizeFeatures()) # just using Cora_p right now
    data = dataset[0]
    data.num_classes = dataset.num_classes
    data = train_test_split_edges_no_neg_adj_mask(data, val_ratio, test_ratio)
    return data

    
def post_process_data(data, args, subset = 'in'):
    data = split_forget_retain(data, args.df_size, subset)
    return data

def main():
    args = parse_args()
    utils.seed_everything(args.random_seed)
    
    print("==CLEAN DATASET==")
    data = get_processed_data(args.dataset)
    args.in_dim = data.x.shape[1]
    train_pos_edge_index = to_undirected(data.train_pos_edge_index)
    data.train_pos_edge_index = train_pos_edge_index
    data.dtrain_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    assert is_undirected(data.train_pos_edge_index)
    print('Undirected dataset:', data)

    if "gnndelete" in args.unlearning_model:
        model = GCNDelete(data.num_features, args.hidden_dim, data.num_classes)
    else:
        model = GCN(data.num_features, args.hidden_dim, data.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    temp = args.unlearning_model
    args.unlearning_model = "original"

    print("==CLEAN TRAINING==")
    trainer = EdgeTrainer(args)
    trainer.train(model, data, optimizer, args)

    print("==CLEAN TESTING==")
    test_results = trainer.test(model, data)
    print(test_results[-1])

    args.unlearning_model = temp
    post_process_data(data, args)

    print("==Membership Inference attack==")
    # Initialize MIAttackTrainer
    mia_trainer = MIAttackTrainer(args)
    # Train shadow model
    all_neg = mia_trainer.train_shadow(model, data, optimizer, args)
    # Prepare attack training data
    feature, label = mia_trainer.prepare_attack_training_data(model, data, leak='posterior', all_neg=all_neg)
    # Create DataLoader for attack model
    train_loader = DataLoader(TensorDataset(feature, label), batch_size=32, shuffle=True)
    valid_loader = DataLoader(TensorDataset(feature, label), batch_size=32)

    attack_model = nn.Sequential(
        nn.Linear(feature.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )
    attack_optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.01, weight_decay=5e-4)
    # Train attack model
    mia_trainer.train_attack(attack_model, train_loader, valid_loader, attack_optimizer, leak='posterior', args=args)
    # Evaluate attack model
    eval_loss, eval_acc, eval_auc, eval_f1 = mia_trainer.eval_attack(attack_model, valid_loader)

    print(f"Attack Model Evaluation - Loss: {eval_loss}, Accuracy: {eval_acc}, AUC: {eval_auc}, F1 Score: {eval_f1}")

    print("==UNLEARNING==")
    if "gnndelete" in args.unlearning_model:
        unlearn_model = GCNDelete(data.num_features, args.hidden_dim, data.num_classes, mask_1hop=data.sdf_node_1hop_mask, mask_2hop=data.sdf_node_2hop_mask)
        
        # copy the weights from the poisoned model
        unlearn_model.load_state_dict(model.state_dict())
        args.unlearning_model = "gnndelete_edge"
        optimizer_unlearn= utils.get_optimizer(args, unlearn_model)
        unlearn_trainer= utils.get_trainer(args, unlearn_model, data, optimizer_unlearn)
        unlearn_trainer.train(unlearn_model, data, optimizer_unlearn, args, attack_model_all=attack_model)
    elif "retrain" in args.unlearning_model:
        unlearn_model = GCN(data.num_features, args.hidden_dim, data.num_classes)
        optimizer_unlearn= utils.get_optimizer(args, unlearn_model)
        unlearn_trainer= utils.get_trainer(args, unlearn_model, data, optimizer_unlearn)
        unlearn_trainer.train()
    else:
        optimizer_unlearn= utils.get_optimizer(args, model)
        unlearn_trainer= utils.get_trainer(args, model, data, optimizer_unlearn)
        unlearn_trainer.train()

if __name__ == "__main__":
    main()
