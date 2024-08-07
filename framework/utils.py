import random
import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph
import os
from torch_geometric.datasets import CitationFull, Coauthor, Amazon, Planetoid, Reddit, Flickr, Twitch
from ogb.linkproppred import PygLinkPropPredDataset
import torch_geometric.transforms as T
from framework.training_args import parse_args

from trainers.contrast import ContrastiveUnlearnTrainer
from trainers.gnndelete import GNNDeleteNodeembTrainer, GNNDeleteEdgeTrainer
from trainers.gnndelete_ni import GNNDeleteNITrainer
from trainers.gradient_ascent import GradientAscentTrainer
from trainers.gif import GIFTrainer, GIFEdgeTrainer
from trainers.base import Trainer
from trainers.utu import UtUTrainer, UtUEdgeTrainer
from trainers.retrain import RetrainTrainer
from trainers.scrub import ScrubTrainer


args = parse_args()

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
        dataset = Reddit(os.path.join(data_dir, d), transform=T.NormalizeFeatures())
    elif d in ['Flickr']:
        dataset = Flickr(os.path.join(data_dir, d), transform=T.NormalizeFeatures())
    elif d in ['Twitch']:
        dataset= Twitch(os.path.join(data_dir, d), name="RU", transform=T.NormalizeFeatures())
    elif 'ogbl' in d:
        dataset = PygLinkPropPredDataset(root=os.path.join(data_dir, d), name=d)
    else:
        raise NotImplementedError(f"{d} not supported.")
    data = dataset[0]
    data.num_classes= dataset.num_classes
    try:
        temp= data.train_mask
    except:
        split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
        data= split(data)
    return data

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def get_sdf_masks(data):
    _, two_hop_edge, _, two_hop_mask = k_hop_subgraph(
        data.edge_index[:, data.df_mask].flatten().unique(),
        2,
        data.edge_index,
        num_nodes=data.num_nodes,
    )
    data.sdf_mask = two_hop_mask

    _, one_hop_edge, _, one_hop_mask = k_hop_subgraph(
        data.edge_index[:, data.df_mask].flatten().unique(),
        1,
        data.edge_index,
        num_nodes=data.num_nodes,
    )
    sdf_node_1hop = torch.zeros(data.num_nodes, dtype=torch.bool)
    sdf_node_2hop = torch.zeros(data.num_nodes, dtype=torch.bool)

    sdf_node_1hop[one_hop_edge.flatten().unique()] = True
    sdf_node_2hop[two_hop_edge.flatten().unique()] = True
    data.sdf_node_1hop_mask = sdf_node_1hop
    data.sdf_node_2hop_mask = sdf_node_2hop
    two_hop_mask = two_hop_mask.bool()
    data.directed_df_edge_index = data.edge_index[:, data.df_mask]
    data.train_pos_edge_index = data.edge_index
    data.sdf_mask = two_hop_mask


def find_masks(data, poisoned_indices, attack_type="label"):
    if attack_type == "label" or attack_type == "random":
        if "scrub" in args.unlearning_model:
            data.df_mask = torch.zeros(data.num_nodes, dtype=torch.bool)  # of size num nodes
            data.dr_mask = data.train_mask
            data.df_mask[poisoned_indices] = True
            data.dr_mask[poisoned_indices] = False
        else:
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
        data.df_mask[poisoned_indices] = True
        data.dr_mask = ~data.df_mask
    data.attacked_idx = poisoned_indices
    if "scrub" in args.unlearning_model:
        return
    get_sdf_masks(data)


def get_trainer(args, poisoned_model, poisoned_data, optimizer_unlearn) -> Trainer:

    trainer_map = {
        "original": Trainer,
        "gradient_ascent": GradientAscentTrainer,
        "gnndelete": GNNDeleteNodeembTrainer,
        "gnndelete_ni": GNNDeleteNITrainer,
        "gnndelete_edge": GNNDeleteEdgeTrainer,
        "gif": GIFTrainer,
        "gif_edge": GIFEdgeTrainer,
        "utu": UtUTrainer,
        "utu_edge": UtUEdgeTrainer,
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
            optimizer_unlearn = [optimizer1, optimizer2]
        else:
            optimizer_unlearn = torch.optim.Adam(parameters_to_optimize, lr=args.unlearn_lr)
    else:
        parameters_to_optimize = [
            {'params': [p for n, p in poisoned_model.named_parameters()], 'weight_decay': args.weight_decay}
        ]
        print('parameters_to_optimize', [n for n, p in poisoned_model.named_parameters()])
        optimizer_unlearn = torch.optim.Adam(parameters_to_optimize, lr=args.unlearn_lr)
    return optimizer_unlearn
