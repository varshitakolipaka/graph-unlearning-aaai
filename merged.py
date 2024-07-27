import copy
import os
import torch
from torch_geometric.seed import seed_everything
from torch_geometric.utils import to_undirected, is_undirected, k_hop_subgraph
from torch_geometric.seed import seed_everything
from framework import get_model, get_trainer
from framework.training_args import parse_args
from framework.data_loader import get_original_data, train_test_split_edges_no_neg_adj_mask, split_forget_retain
from framework.trainer.label_poison import LabelPoisonTrainer
from framework.trainer.edge_poison import EdgePoisonTrainer
from framework.trainer.label_poison import get_label_poisoned_data
from framework.trainer.edge_poison import get_edge_poisoned_data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_processed_data(d, val_ratio=0.05, test_ratio=0.05):
    data = get_original_data(d)
    data = train_test_split_edges_no_neg_adj_mask(data, val_ratio, test_ratio)
    return data

def get_data_attack(args):
    data = get_original_data(args.dataset)
    return data

def get_processed_data_unlearn(args, val_ratio, test_ratio, df_ratio, subset='in'):
    '''pend for future use'''
    data = get_original_data(args.dataset)
    if args.request == 'edge':
        data = train_test_split_edges_no_neg_adj_mask(data, val_ratio, test_ratio)
        data = split_forget_retain(data, df_ratio, subset)
    elif args.attack_type=="label":
        data, flipped_indices = get_label_poisoned_data(args, data, df_ratio, args.random_seed)
        data.df_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
        data.dr_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
        for node in flipped_indices:
            data.train_mask[node] = False
            node_tensor = torch.tensor([node], dtype=torch.long)
            _, local_edges, _, mask = k_hop_subgraph(
                node_tensor, 1, data.edge_index, num_nodes=data.num_nodes)
            data.df_mask[mask] = True
        data.dr_mask = ~data.df_mask
    elif args.attack_type=="edge":
        augmented_edges, poisoned_indices = get_edge_poisoned_data(args, data, df_ratio, args.random_seed)
        data.edge_index= augmented_edges
        data.df_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
        data.dr_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)

        data.df_mask[poisoned_indices] = True
        data.dr_mask = ~data.df_mask
    return data


def train():
    args = parse_args()
    print(args)
    args.unlearning_model = 'original_node'
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model, str(args.random_seed))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    seed_everything(args.random_seed)
    data = get_processed_data(args.dataset)
    if args.gnn not in ['rgcn', 'rgat']:
        args.in_dim = data.x.shape[1]
    if args.gnn in ['rgcn', 'rgat']:
        if not hasattr(data, 'train_mask'):
            data.train_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)
    else:
        data.dtrain_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    train_pos_edge_index = to_undirected(data.train_pos_edge_index)
    data.train_pos_edge_index = train_pos_edge_index
    data.dtrain_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    assert is_undirected(data.train_pos_edge_index)
    model = get_model(args, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=args.weight_decay)
    trainer = get_trainer(args)
    print(trainer)
    trainer.train(model, data, optimizer, args)
    test_results = trainer.test(model, data)
    trainer.save_log()
    print(test_results[-1])
    return model, data


def attack(model=None):
    args = parse_args()
    original_path = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'original_node', str(args.random_seed))
    shadow_path_all = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'shadow_all', str(args.random_seed))
    args.shadow_dir = shadow_path_all
    if not os.path.exists(shadow_path_all):
        os.makedirs(shadow_path_all, exist_ok=True)
    args.checkpoint_dir = os.path.join(
        args.checkpoint_dir, args.dataset, args.gnn, args.attack_type,
        '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    seed_everything(args.random_seed)

    data = get_data_attack(args)
    if(args.attack_type=="label"):
        args.unlearning_model = 'lf_attack'
        lp_trainer = LabelPoisonTrainer(args)
        data, flipped_indices = lp_trainer.label_flip_attack(data, args.df_size, args.random_seed)
    elif(args.attack_type=="edge"):
        args.unlearning_model = 'edge_attack'
        edge_trainer= EdgePoisonTrainer(args)
        aug, poisoned_indices, poisoned_nodes= edge_trainer.edge_attack_random_nodes(data, args.df_size, args.random_seed)
        data.poisoned_nodes= poisoned_nodes
        data.edge_index= aug
    print(data)
    if args.gnn not in ['rgcn', 'rgat']:
        args.in_dim = data.x.shape[1]
    if model is None:
        model = get_model(args)
    else:
        model= copy.deepcopy(model)

    parameters_to_optimize = [
        {'params': [p for n, p in model.named_parameters()], 'weight_decay': args.weight_decay}
    ]
    print('parameters_to_optimize', [n for n, p in model.named_parameters()])
    optimizer = torch.optim.Adam(parameters_to_optimize, lr=args.lr)
    trainer = get_trainer(args)
    print(trainer)
    model = model.to(device)
    trainer.train(model, data, optimizer, args)
    test_results = trainer.test(model, data)
    trainer.save_log()
    print(test_results[-1])
    return model, data

torch.autograd.set_detect_anomaly(True)
def unlearn(model=None):
    args = parse_args()
    original_path = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, args.attack_type, 'in-' + str(args.df_size) + '-' + str(args.random_seed))
    attack_path_all = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'member_infer_all', str(args.random_seed))
    args.attack_dir = attack_path_all
    if not os.path.exists(attack_path_all):
        os.makedirs(attack_path_all, exist_ok=True)
    shadow_path_all = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, 'shadow_all', str(args.random_seed))
    args.shadow_dir = shadow_path_all
    if not os.path.exists(shadow_path_all):
        os.makedirs(shadow_path_all, exist_ok=True)

    args.checkpoint_dir = os.path.join(
        args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model,
        '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]))
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    seed_everything(args.random_seed)
    data = get_processed_data_unlearn(args, val_ratio=0.05, test_ratio=0.05, df_ratio=args.df_size)
    print('Directed dataset:', data)
    if args.gnn not in ['rgcn', 'rgat']:
        args.in_dim = data.x.shape[1]

    if model is None:
        model = get_model(args, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)
        if args.unlearning_model != 'retrain':  # Start from trained GNN model
            if os.path.exists(os.path.join(original_path, 'pred_proba.pt')):
                logits_ori = torch.load(os.path.join(original_path, 'pred_proba.pt'))   # logits_ori: tensor.shape([num_nodes, num_nodes]), represent probability of edge existence between any two nodes
                if logits_ori is not None:
                    logits_ori = logits_ori.to(device)
            else:
                logits_ori = None

            model_ckpt = torch.load(os.path.join(original_path, 'model_best.pt'), map_location=device)
            model.load_state_dict(model_ckpt['model_state'], strict=False)

    else:       # Initialize a new GNN model
        retrain = None
        logits_ori = None

    model = model.to(device)
    if 'gnndelete' in args.unlearning_model:
        parameters_to_optimize = [
            {'params': [p for n, p in model.named_parameters() if 'del' in n], 'weight_decay': args.weight_decay}
        ]
        print('parameters_to_optimize', [n for n, p in model.named_parameters() if 'del' in n])
        if 'layerwise' in args.loss_type:
            optimizer1 = torch.optim.Adam(model.deletion1.parameters(), lr=args.lr)
            optimizer2 = torch.optim.Adam(model.deletion2.parameters(), lr=args.lr)
            optimizer = [optimizer1, optimizer2]
        else:
            optimizer = torch.optim.Adam(parameters_to_optimize, lr=args.lr)
    else:
        parameters_to_optimize = [
            {'params': [p for n, p in model.named_parameters()], 'weight_decay': args.weight_decay}
        ]
        print('parameters_to_optimize', [n for n, p in model.named_parameters()])
        optimizer = torch.optim.Adam(parameters_to_optimize, lr=args.lr)
    trainer = get_trainer(args)
    print('Trainer: ', trainer)
    print(f"df mask: {data.df_mask.sum().item()}") # 5702
    print(f"dr mask: {data.dr_mask.sum().item()}") # 108452 -> are these edges?
    print(data.df_mask.sum().item() + data.dr_mask.sum().item() == data.edge_index.size(1)) # True
    print(f"length of data.x: {data.x.size(dim=0)}") # The length of the x is 19763.
    unlearnt_model = copy.deepcopy(model)
    trainer.train(unlearnt_model, data, optimizer, args)
    if args.unlearning_model != 'retrain':
        retrain_path = os.path.join(
            'checkpoint', args.dataset, args.gnn, 'retrain',
            '-'.join([str(i) for i in [args.df, args.df_size, args.random_seed]]),
            'model_best.pt')
        if os.path.exists(retrain_path):
            retrain_ckpt = torch.load(retrain_path, map_location=device)
            retrain_args = copy.deepcopy(args)
            retrain_args.unlearning_model = 'retrain'
            retrain = get_model(retrain_args, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type)
            retrain.load_state_dict(retrain_ckpt['model_state'])
            retrain = retrain.to(device)
            retrain.eval()
        else:
            retrain = None
    else:
        retrain = None
    test_results = trainer.test(unlearnt_model, data, model_retrain=None, attack_model_all=None, attack_model_sub=None, is_dr = True)
    print(test_results[-1])
    trainer.save_log()
    return unlearnt_model

def cal_GSL(model, graph, node_list):
    with torch.no_grad():
        x= model(graph.x, graph.edge_index)
        x= x[node_list].to("cpu")
        num_nodes = x.shape[0]
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        batch_size = 1000 if num_nodes > 1000 else num_nodes
        for i in range(0, num_nodes, batch_size):
            now_node_SL = cos(
                x.unsqueeze(0), x[i : i + batch_size].unsqueeze(1)
            ).fill_diagonal_(0).sum(1) / float(num_nodes - 1.0)
            if i == 0:
                node_SL = now_node_SL
            else:
                node_SL = torch.cat(
                    [
                        node_SL,
                        now_node_SL,
                    ],
                )
        GSL = node_SL.sum() / num_nodes
    return GSL.item()

if __name__ == "__main__":
    clean_model, clean_data= train()
    original_model= copy.deepcopy(clean_model)
    original_data= copy.deepcopy(clean_data)

    poisoned_model, poisoned_data= attack(clean_model)
    unlearnt_model= unlearn(poisoned_model)
    
    a= cal_GSL(original_model, original_data, poisoned_data.poisoned_nodes)
    b= cal_GSL(poisoned_model, poisoned_data, poisoned_data.poisoned_nodes)
    c= cal_GSL(unlearnt_model, poisoned_data, poisoned_data.poisoned_nodes)
    print(a, b, c)