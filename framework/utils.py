import random
import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph
import os
from torch_geometric.datasets import CitationFull, Coauthor, Amazon, Planetoid, Reddit2, Flickr, Twitch
import torch_geometric.transforms as T

from trainers.contrast import ContrastiveUnlearnTrainer
from trainers.gnndelete import GNNDeleteNodeembTrainer
from trainers.gnndelete_ni import GNNDeleteNITrainer
from trainers.gradient_ascent import GradientAscentTrainer
from trainers.gif import GIFTrainer
from trainers.base import Trainer
from trainers.scrub import ScrubTrainer
from trainers.utu import UtUTrainer
from trainers.retrain import RetrainTrainer

def get_original_data(d):
    data_dir = './datasets'
    if d in ['Cora', 'PubMed', 'DBLP']:
        dataset = CitationFull(os.path.join(data_dir, d), d, transform=T.NormalizeFeatures())
    elif d in ['Cora_p', 'PubMed_p', 'Citeseer_p']:
        dataset = Planetoid(os.path.join(data_dir, d), d.split('_')[0], transform=T.NormalizeFeatures())
    elif d in ['CS', 'Physics']:
        dataset = Coauthor(os.path.join(data_dir, d), d, transform=T.NormalizeFeatures())
    elif d in ['Amazon']:
        dataset = Amazon(os.path.join(data_dir, d), 'Photo', transform=T.NormalizeFeatures())
    elif d in ['Reddit']:
        dataset = Reddit2(os.path.join(data_dir, d), transform=T.NormalizeFeatures())
    elif d in ['Flickr']:
        dataset = Flickr(os.path.join(data_dir, d), transform=T.NormalizeFeatures())
    elif d in ['Twitch']:
        dataset = Twitch(os.path.join(data_dir, d), "EN", transform=T.NormalizeFeatures())
    else:
        raise NotImplementedError(f"{d} not supported.")
    data = dataset[0]

    data.num_classes= dataset.num_classes
    return data

def train_test_split(data, seed, train_ratio=0.1):
    n = data.num_nodes
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)
    train_idx = idx[:int(n*train_ratio)]
    test_idx = idx[int(n*train_ratio):]
    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.test_mask = torch.zeros(n, dtype=torch.bool)
    data.test_mask[test_idx] = True
    return data

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def get_sdf_masks(data, args):
    if args.attack_type!="edge":
        _, three_hop_edge, _, three_hop_mask = k_hop_subgraph(
            data.edge_index[:, data.df_mask].flatten().unique(),
            3,
            data.edge_index,
            num_nodes=data.num_nodes,
        )
        _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
            data.edge_index[:, data.df_mask].flatten().unique(),
            2,
            data.edge_index,
            num_nodes=data.num_nodes,
        )
        _, one_hop_edge, _, one_hop_mask = k_hop_subgraph(
            data.edge_index[:, data.df_mask].flatten().unique(),
            1,
            data.edge_index,
            num_nodes=data.num_nodes,
        )
    else:
        _, three_hop_edge, _, three_hop_mask = k_hop_subgraph(
            data.poisoned_nodes,
            3,
            data.edge_index,
            num_nodes=data.num_nodes,
        )
        _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
            data.poisoned_nodes,
            2,
            data.edge_index,
            num_nodes=data.num_nodes,
        )
        _, one_hop_edge, _, one_hop_mask = k_hop_subgraph(
            data.poisoned_nodes,
            1,
            data.edge_index,
            num_nodes=data.num_nodes,
        )
    data.sdf_mask = two_hop_mask
    sdf_node_1hop = torch.zeros(data.num_nodes, dtype=torch.bool)
    sdf_node_2hop = torch.zeros(data.num_nodes, dtype=torch.bool)
    sdf_node_3hop = torch.zeros(data.num_nodes, dtype=torch.bool)

    sdf_node_1hop[one_hop_edge.flatten().unique()] = True
    sdf_node_2hop[two_hop_edge.flatten().unique()] = True
    sdf_node_3hop[three_hop_edge.flatten().unique()] = True

    data.sdf_node_1hop_mask = sdf_node_1hop
    data.sdf_node_2hop_mask = sdf_node_2hop
    data.sdf_node_3hop_mask = sdf_node_3hop

    three_hop_mask = three_hop_mask.bool()
    data.directed_df_edge_index = data.edge_index[:, data.df_mask]
    data.train_pos_edge_index = data.edge_index
    data.sdf_mask = three_hop_mask


def find_masks(data, poisoned_indices, args, attack_type="label"):
    if attack_type == "label" or attack_type == "random"  or attack_type == "trigger":
        data.df_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
        data.dr_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
        for node in poisoned_indices:
            data.train_mask[node] = False
            node_tensor = torch.tensor([node], dtype=torch.long)
            _, local_edges, _, mask = k_hop_subgraph(
                node_tensor, 1, data.edge_index, num_nodes=data.num_nodes
            )
            data.df_mask[mask] = True
        data.dr_mask = ~data.df_mask
    elif attack_type == "edge":
        data.df_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
        data.dr_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
        data.df_mask[poisoned_indices] = 1
        data.dr_mask = ~data.df_mask
    data.attacked_idx = poisoned_indices
    get_sdf_masks(data, args)


def get_trainer(args, poisoned_model, poisoned_data, optimizer_unlearn) -> Trainer:

    trainer_map = {
        "original": Trainer,
        "gradient_ascent": GradientAscentTrainer,
        "gnndelete": GNNDeleteNodeembTrainer,
        "gnndelete_ni": GNNDeleteNITrainer,
        "gif": GIFTrainer,
        "utu": UtUTrainer,
        "contrastive": ContrastiveUnlearnTrainer,
        "retrain": RetrainTrainer,
        "scrub": ScrubTrainer
    }

    if args.unlearning_model in trainer_map:
        return trainer_map[args.unlearning_model](poisoned_model, poisoned_data, optimizer_unlearn, args)
    else:
        raise NotImplementedError(f"{args.unlearning_model} not implemented yet")

def get_optimizer(args, poisoned_model):
    if 'gnndelete' in args.unlearning_model:
        parameters_to_optimize = [
            {'params': [p for n, p in poisoned_model.named_parameters() if 'del' in n], 'weight_decay': args.weight_decay}
        ]
        print('parameters_to_optimize', [n for n, p in poisoned_model.named_parameters() if 'del' in n])
        if 'layerwise' in args.loss_type:
            optimizer1 = torch.optim.Adam(poisoned_model.deletion1.parameters(), lr=args.unlearn_lr)
            optimizer2 = torch.optim.Adam(poisoned_model.deletion2.parameters(), lr=args.unlearn_lr)
            optimizer3 = torch.optim.Adam(poisoned_model.deletion3.parameters(), lr=args.unlearn_lr)
            optimizer_unlearn = [optimizer1, optimizer2, optimizer3]
        else:
            optimizer_unlearn = torch.optim.Adam(parameters_to_optimize, lr=args.unlearn_lr)
    else:
        parameters_to_optimize = [
            {'params': [p for n, p in poisoned_model.named_parameters()], 'weight_decay': args.weight_decay}
        ]
        print('parameters_to_optimize', [n for n, p in poisoned_model.named_parameters()])
        optimizer_unlearn = torch.optim.Adam(parameters_to_optimize, lr=args.unlearn_lr)
    return optimizer_unlearn

def prints_stats(data):
    # print the stats of the dataset
    print("Number of nodes: ", data.num_nodes)
    print("Number of edges: ", data.num_edges)
    print("Number of features: ", data.num_features)
    print("Number of classes: ", data.num_classes)
    print("Number of training nodes: ", data.train_mask.sum().item())
    print("Number of testing nodes: ", data.test_mask.sum().item())