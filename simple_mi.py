import torch
import os
import math
import optuna
import torch.nn as nn
import torch_geometric.transforms as T
import torch.optim as optim
import copy
from framework.training_args import parse_args
from framework import utils
from models.deletion import GCNDelete
from models.models import GCN
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import is_undirected, to_undirected, negative_sampling, k_hop_subgraph
from trainers.base import EdgeTrainer
from attacks.mi_attack import MIAttackTrainer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from attacks.mi_attack import AttackModel
from trainers.base import member_infer_attack
# import randomforest from sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
args = parse_args()
utils.seed_everything(args.random_seed)

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
    # data.edge_index = train_pos_edge_index
    assert is_undirected(data.train_pos_edge_index)

    data.directed_df_edge_index = data.train_pos_edge_index[:, df_mask]
    data.sdf_mask = two_hop_mask
    data.df_mask = df_mask
    data.dr_mask = dr_mask

    # Identify poisoned indices (edges that are removed)
    poisoned_indices = df_global_idx
    data.attacked_idx = poisoned_indices


    poisoned_nodes = torch.unique(data.train_pos_edge_index[:, df_global_idx].flatten()).tolist()
    data.poisoned_nodes = poisoned_nodes

    return data

def get_processed_data(d, val_ratio=0.05, test_ratio=0.35):
    data_dir = './data' 
    dataset = Planetoid(os.path.join(data_dir, d), d.split('_')[0], transform=T.NormalizeFeatures()) # just using Cora_p right now
    data = dataset[0]
    data.num_classes = dataset.num_classes
    data = train_test_split_edges_no_neg_adj_mask(data, val_ratio, test_ratio)
    return data
    
def post_process_data(data, args, subset = 'in'):
    data = split_forget_retain(data, args.df_size, subset)
    return data

# Function to get logits from the model
def get_logits(model, data, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model(data.x, pos_edge_index)  # Pass edge_index to the model
        logits = model.dup_decode(z, pos_edge_index, neg_edge_index)
    return logits

def report_logits_ratio(df_mask, logits_before, logits_after):
    # Compute mean ratio of logits before and after unlearning
    logits_before_df_mask = logits_before[df_mask]
    logits_after_df_mask = logits_after[df_mask]
    
    ratios = logits_after_df_mask/ logits_before_df_mask
    mean_ratio = ratios.mean().item()
    
    print(f'Mean Logits Ratio (Before/After Unlearning): {mean_ratio:.4f}')
    return mean_ratio

def main():
    print("==CLEAN DATASET==")
    data = get_processed_data(args.dataset)
    
    args.in_dim = data.x.shape[1]
    
    train_pos_edge_index = to_undirected(data.train_pos_edge_index)
    data.train_pos_edge_index = train_pos_edge_index
    data.val_pos_edge_index = to_undirected(data.val_pos_edge_index)
    data.test_pos_edge_index = to_undirected(data.test_pos_edge_index)

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
    data = post_process_data(data, args) # to get df, dr, sdf masks

    print("====================")
    print('Train Size: ', data.train_pos_edge_index.shape)
    print('Val Size: ', data.val_pos_edge_index.shape)
    print('Test Size: ', data.test_pos_edge_index.shape)
    print('Edge Index:', data.edge_index.shape)
    print('Node Features:', data.x.shape)
    print('Number Features:', data.num_features)

    print('Mask Sizes: ', data.df_mask.sum(), data.dr_mask.sum())
    print("====================")

    print("==Membership Inference attack==") ## possibly wrong
    # Initialize MIAttackTrainer
    attack_model = AttackModel(data.num_classes)

    pos_edge_index = data.train_pos_edge_index

    # Negative edges from the validation and test sets
    neg_edge_index = torch.cat([data.val_pos_edge_index, data.test_pos_edge_index], dim=1)

    # Combine pos and neg edges
    all_edges = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    logits = get_logits(model, data, pos_edge_index, neg_edge_index).to(device)

    # Create labels: 1 for positive edges, 0 for negative edges
    labels = torch.cat([torch.ones(pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))])

    # Split data into training and validation sets (80-20 split)
    train_logits, valid_logits, train_labels, valid_labels = train_test_split(logits.cpu().numpy(), labels.cpu().numpy(), test_size=0.2, random_state=args.random_seed)
    train_logits = torch.tensor(train_logits, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.float32).to(device)
    valid_labels = torch.tensor(valid_labels, dtype=torch.float32).to(device)
    valid_logits = torch.tensor(valid_logits, dtype=torch.float32).to(device)

    tsne = TSNE(n_components=2, random_state=0)
    logits_tsne = tsne.fit_transform(logits.cpu().numpy())

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(logits_tsne[:, 0], logits_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Labels')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE of Logits Colored by Labels')
    plt.savefig('tsne.png')

    # Step 4: Train the Attack Model
    criterion = nn.BCELoss()
    optimizer = optim.Adam(attack_model.parameters(), lr=0.5)

    num_epochs = 1000  # Example, you can adjust this

    for epoch in range(num_epochs):
        attack_model.train()
        optimizer.zero_grad()

        # Forward pass
        attack_model = attack_model.to(device)
        
        outputs = attack_model(train_logits).squeeze()
        outputs = outputs.to(device)
        train_labels = train_labels.to(device)
        loss = criterion(outputs, train_labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Step 5: Evaluate the Attack Model (optional)
    attack_model.eval()
    with torch.no_grad():
        predictions = attack_model(valid_logits).squeeze()
        # You can use metrics like accuracy, precision, recall, etc.
        accuracy = ((predictions > 0.5) == valid_labels).float().mean()
        print(f'Attack Model Accuracy: {accuracy:.4f}')



    # rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # rf.fit(logits.cpu().numpy().reshape(-1, 1), labels.cpu().numpy())
    # # give accruacy of the random forest model
    # print(f'Random Forest Accuracy: {rf.score(logits.cpu().numpy().reshape(-1, 1), labels.cpu().numpy())}')


    exit()

    # if "gnndelete" in args.unlearning_model:
    #     shadow_model = GCNDelete(data.num_features, args.hidden_dim, data.num_classes)
    # else:
    #     shadow_model = GCN(data.num_features, args.hidden_dim, data.num_classes)
    # shadow_optimizer = torch.optim.Adam(shadow_model.parameters(), lr=0.01, weight_decay=5e-4)

    # # Train shadow model
    # all_neg = mia_trainer.train_shadow(shadow_model, data, shadow_optimizer, args)
    # # Prepare attack training data
    # feature, label = mia_trainer.prepare_attack_training_data(shadow_model, data, leak='posterior', all_neg=all_neg)
    # # Create DataLoader for attack model
    
    # # do a 80-20 train test split of feature, label and use DataLoader to load it in train_loader and valid_loader
    # feature_tensor = torch.tensor(feature, dtype=torch.float32)
    # label_tensor = torch.tensor(label, dtype=torch.long)

    # # Split data into training and validation sets (80-20 split)
    # train_features, valid_features, train_labels, valid_labels = train_test_split(feature_tensor, label_tensor, test_size=0.2, random_state=args.random_seed)

    # # Create TensorDataset for training and validation sets
    # train_dataset = TensorDataset(train_features, train_labels)
    # valid_dataset = TensorDataset(valid_features, valid_labels)

    # # Create DataLoader for training and validation sets
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    # # unsure what attack model here means does for the optimizer
    # attack_model = AttackModel(feature.shape[1])

    # attack_optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.01, weight_decay=5e-4)
    # # Train attack model
    # mia_trainer.train_attack(attack_model, train_loader, valid_loader, attack_optimizer, leak='posterior', args=args)

    # logits_before, _ = member_infer_attack(model, attack_model, data, args, before=True)

    # print("Membership Inference Attack before unlearning: ", logits_before)
    print("==UNLEARNING==")
    if "gnndelete" in args.unlearning_model:
        unlearn_model = GCNDelete(data.num_features, args.hidden_dim, data.num_classes, mask_1hop=data.sdf_node_1hop_mask, mask_2hop=data.sdf_node_2hop_mask)
        
        # copy the weights from the poisoned model
        unlearn_model.load_state_dict(model.state_dict())
        optimizer_unlearn = utils.get_optimizer(args, unlearn_model)
        unlearn_trainer= utils.get_trainer(args, unlearn_model, data, optimizer_unlearn)
        unlearn_trainer.train(unlearn_model, data, optimizer_unlearn, args)
        test_results = unlearn_trainer.test(unlearn_model, data)
    elif "retrain" in args.unlearning_model:
        unlearn_model = GCN(data.num_features, args.hidden_dim, data.num_classes)
        optimizer_unlearn= utils.get_optimizer(args, unlearn_model)
        unlearn_trainer= utils.get_trainer(args, unlearn_model, data, optimizer_unlearn)
        unlearn_trainer.train(unlearn_model, data, optimizer_unlearn, args)
        test_results = unlearn_trainer.test(unlearn_model, data)
    elif "utu" in args.unlearning_model:
        optimizer_unlearn= utils.get_optimizer(args, model)
        unlearn_trainer= utils.get_trainer(args, model, data, optimizer_unlearn)
        unlearn_trainer.train(model, data, optimizer_unlearn, args)
        test_results = unlearn_trainer.test(model, data)
    elif "gif" in args.unlearning_model:
        optimizer_unlearn= utils.get_optimizer(args, model)
        unlearn_trainer= utils.get_trainer(args, model, data, optimizer_unlearn)
        unlearn_trainer.train(model, data, optimizer_unlearn, args)
        test_results = unlearn_trainer.test(model, data)
    elif "contrastive" in args.unlearning_model:
        args.request="edge"
        print("Number of edges removed: ", len(data.attacked_idx))
        optimizer_unlearn= utils.get_optimizer(args, model)
        unlearn_trainer= utils.get_trainer(args, model, data, optimizer_unlearn)
        unlearn_trainer.train(attack_model=attack_model)
        test_results = unlearn_trainer.test(model, data)
    elif "gradient_ascent" in args.unlearning_model:
        optimizer_unlearn= utils.get_optimizer(args, model)
        unlearn_trainer= utils.get_trainer(args, model, data, optimizer_unlearn)
        unlearn_trainer.train(model, data, optimizer_unlearn, args)
        test_results = unlearn_trainer.test(model, data)
        
    print("==CLEAN TESTING AFTER UNLEARNING==")
    
    print(test_results[-1])

    print("DT_AUC: ", test_results[1])

    mi_score = report_logits_ratio(data.df_mask, logits)

if __name__ == "__main__":
    main()