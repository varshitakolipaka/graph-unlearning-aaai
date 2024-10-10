import random
from models.deletion import GATDelete, GCNDelete, GINDelete
from models.models import GAT, GCN, GIN
import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph
import os
from torch_geometric.datasets import CitationFull, Coauthor, Amazon, Planetoid, Reddit2, Flickr, Twitch
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch_geometric.utils import subgraph


from trainers.contrascent import ContrastiveAscentTrainer
from trainers.contrascent_no_link import ContrastiveAscentNoLinkTrainer
from trainers.contrast import ContrastiveUnlearnTrainer
from trainers.contrast_another import ContrastiveUnlearnTrainer_NEW
from trainers.gnndelete import GNNDeleteNodeembTrainer
from trainers.gnndelete_ni import GNNDeleteNITrainer
from trainers.gradient_ascent import GradientAscentTrainer
from trainers.gif import GIFTrainer
from trainers.base import Trainer
from trainers.scrub import ScrubTrainer
from trainers.scrub_no_kl import ScrubTrainer1
from trainers.scrub_no_kl_combined import ScrubTrainer2
from trainers.ssd import SSDTrainer
from trainers.utu import UtUTrainer
from trainers.retrain import RetrainTrainer
from trainers.megu import MeguTrainer
from trainers.grub import GrubTrainer
from trainers.yaum import YAUMTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_original_data(d):
    data_dir = './datasets'
    if d in ['Cora', 'PubMed', 'DBLP', 'Cora_ML']:
        dataset = CitationFull(os.path.join(data_dir, d), d, transform=T.NormalizeFeatures())
    elif d in ['Cora_p', 'PubMed_p', 'Citeseer_p']:
        dataset = Planetoid(os.path.join(data_dir, d), d.split('_')[0], transform=T.NormalizeFeatures())
    elif d in ['CS', 'Physics']:
        dataset = Coauthor(os.path.join(data_dir, d), d, transform=T.NormalizeFeatures())
    elif d in ['Amazon']:
        dataset = Amazon(os.path.join(data_dir, d), 'Photo', transform=T.NormalizeFeatures())
    elif d in ['Computers']:
        dataset = Amazon(os.path.join(data_dir, d), 'Computers', transform=T.NormalizeFeatures())
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
    transform = T.LargestConnectedComponents()
    data = transform(data)
    return data

def get_model(args, in_dim, hidden_dim, out_dim, mask_1hop=None, mask_2hop=None, mask_3hop=None):

    if 'gnndelete' in args.unlearning_model:
        model_mapping = {'gcn': GCNDelete, 'gat': GATDelete, 'gin': GINDelete}
        return model_mapping[args.gnn](in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, mask_1hop=mask_1hop, mask_2hop=mask_2hop, mask_3hop=mask_3hop)

    else:
        model_mapping = {'gcn': GCN, 'gat': GAT, 'gin': GIN}
        return model_mapping[args.gnn](in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)


def train_test_split(data, seed, train_ratio=0.1, val_ratio=0.1):
    n = data.num_nodes
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)
    train_idx = idx[:int(train_ratio * n)]
    val_idx = idx[int(train_ratio * n):int((train_ratio + val_ratio) * n)]
    test_idx = idx[int((train_ratio + val_ratio) * n):]
    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.val_mask = torch.zeros(n, dtype=torch.bool)
    data.test_mask = torch.zeros(n, dtype=torch.bool)

    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True

    return data, train_idx, test_idx

def inductive_graph_split(data):
    train_edge_index, _ = subgraph(data.train_mask, data.edge_index)
    data.edge_index = train_edge_index

    val_edge_index, _ = subgraph(data.val_mask, data.edge_index)
    data.val_edge_index = val_edge_index

    test_edge_index, _ = subgraph(data.test_mask, data.edge_index)
    data.test_edge_index = test_edge_index


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

        if "scrub" in args.unlearning_model or "grub" in args.unlearning_model or "yaum" in args.unlearning_model or "ssd" in args.unlearning_model or ("megu" in args.unlearning_model and "node" in args.request):
            data.node_df_mask = torch.zeros(data.num_nodes, dtype=torch.bool)  # of size num nodes
            data.node_dr_mask = data.train_mask
            data.node_df_mask[poisoned_indices] = True
            data.node_dr_mask[poisoned_indices] = False

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
    data.attacked_idx = torch.tensor(poisoned_indices, dtype=torch.long)
    if not ("scrub" in args.unlearning_model) and not ("megu" in args.unlearning_model):
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
        'contra_2': ContrastiveUnlearnTrainer_NEW,
        "retrain": RetrainTrainer,
        "scrub": ScrubTrainer,
        "megu": MeguTrainer,
        "ssd": SSDTrainer,
        "grub": GrubTrainer,
        "yaum": YAUMTrainer,
        "contrascent": ContrastiveAscentTrainer,
        'cacdc': ContrastiveAscentNoLinkTrainer,
        "scrub_no_kl": ScrubTrainer1,
        "scrub_no_kl_combined": ScrubTrainer2
    }

    if args.unlearning_model in trainer_map:
        return trainer_map[args.unlearning_model](poisoned_model, poisoned_data, optimizer_unlearn, args)
    else:
        raise NotImplementedError(f"{args.unlearning_model} not implemented yet")

def get_optimizer(args, poisoned_model):
    if 'gnndelete' in args.unlearning_model:
        parameters_to_optimize = [
            {'params': [p for n, p in poisoned_model.named_parameters() if 'del' in n]}
        ]
        print('parameters_to_optimize', [n for n, p in poisoned_model.named_parameters() if 'del' in n])
        if 'layerwise' in args.loss_type:
            optimizer1 = torch.optim.Adam(poisoned_model.deletion1.parameters(), lr=args.unlearn_lr, weight_decay=args.weight_decay)
            optimizer2 = torch.optim.Adam(poisoned_model.deletion2.parameters(), lr=args.unlearn_lr, weight_decay=args.weight_decay)
            optimizer3 = torch.optim.Adam(poisoned_model.deletion3.parameters(), lr=args.unlearn_lr, weight_decay=args.weight_decay)
            optimizer_unlearn = [optimizer1, optimizer2, optimizer3]
        else:
            optimizer_unlearn = torch.optim.Adam(parameters_to_optimize, lr=args.unlearn_lr, weight_decay=args.weight_decay)
    elif 'retrain' in args.unlearning_model:
        optimizer_unlearn = torch.optim.Adam(poisoned_model.parameters(), lr=args.unlearn_lr, weight_decay=args.weight_decay)
    else:
        parameters_to_optimize = [
            {'params': [p for n, p in poisoned_model.named_parameters()]}
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

    # get counts of each class
    counts = [0] * data.num_classes
    for i in range(data.num_classes):
        counts[i] = (data.y == i).sum().item()
    for i in range(data.num_classes):
        print(f"Number of nodes in class {i}: {counts[i]}")

def plot_embeddings(args, model, data, class1, class2, is_dr=False, mask="test", name=""):
    # Set the model to evaluation mode
    model.eval()

    # Forward pass: get embeddings
    with torch.no_grad():
        if is_dr and args.unlearning_model != "scrub":
            embeddings = model(data.x, data.edge_index[:, data.dr_mask])
        else:
            embeddings = model(data.x, data.edge_index)

    # If embeddings have more than 2 dimensions, apply t-SNE
    print("Embeddings shape:", embeddings.shape)
    if embeddings.shape[1] > 2:
        embeddings = TSNE(n_components=2).fit_transform(embeddings.cpu())
        embeddings = torch.tensor(embeddings).to(device)
    print("Embeddings shape after t-SNE:", embeddings.shape)
    # Get the mask (either test, train, or val)
    if mask == "test":
        mask = data.test_mask
    elif mask == "train":
        mask = data.train_mask
    else:
        mask = data.val_mask

    # Filter embeddings and labels based on the mask
    embeddings = embeddings[mask]
    labels = data.y[mask]

    # Create masks for class1, class2, and other classes
    class1_mask = (labels == class1)
    class2_mask = (labels == class2)

    # convert masks to numpy
    class1_mask = class1_mask.cpu().numpy()
    class2_mask = class2_mask.cpu().numpy()

    # give a color map to the other classes
    class_masks = {}
    for i in range(data.num_classes):
        if i != class1 and i != class2:
            class_masks[i] = (labels == i).cpu().numpy()

    colors = sns.color_palette("dark", data.num_classes)
    color_map = {}
    for i in range(data.num_classes):
        if i == class1:
            color_map[i] = 'blue'
        elif i == class2:
            color_map[i] = 'red'
        else:
            color_map[i] = colors[i]

    # Prepare the plot
    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")

    # convert to numpy
    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()

    # Plot class1
    plt.scatter(embeddings[class1_mask, 0], embeddings[class1_mask, 1], label=f'Class {class1}', color='blue', alpha=0.6)
    # Plot class2
    plt.scatter(embeddings[class2_mask, 0], embeddings[class2_mask, 1], label=f'Class {class2}', color='red', alpha=0.6)
    # Plot other classes
    for i in class_masks:
        plt.scatter(embeddings[class_masks[i], 0], embeddings[class_masks[i], 1], label=f'Class {i}', color=color_map[i], alpha=0.3)

    # Add legend to top right
    plt.legend(loc='upper right')
    plt.title(f"{args.dataset}-{name}")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")

    # Save the plot
    os.makedirs("./plots", exist_ok=True)
    plt.savefig(f"./plots/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_{name}_embeddings.png")
    plt.show()

def sample_poison_data(poisoned_indices, frac):
    assert frac <= 1.0, "frac must be between 0 and 1"
    if frac <= 0.0:
        num_to_sample = 1
    else:
        num_to_sample = int(frac * len(poisoned_indices))

    sampled_nodes =  torch.tensor(np.random.choice(poisoned_indices.cpu().numpy(), num_to_sample, replace=False))
    print("Sampling for Corrective")

    print(sampled_nodes)
    # exit(0)
    return sampled_nodes

def sample_poison_data_edges(data, frac):
    assert 0.0 <= frac <= 1.0, "frac must be between 0 and 1"
    poisoned_indices = data.poisoned_edge_indices.cpu().numpy()
    
    total_edges_poisoned = len(poisoned_indices) // 2
    num_to_sample = int(frac * total_edges_poisoned)
    
    edge_dict = {}
    unique_edges = []
    cnt = 0
    for i in poisoned_indices:
        edge = (data.edge_index[0][i].item(), data.edge_index[1][i].item())
        reverse_edge = (data.edge_index[1][i].item(), data.edge_index[0][i].item())

        if edge not in edge_dict and reverse_edge not in edge_dict:
            unique_edges.append(i)

        if reverse_edge in edge_dict:
            cnt += 1
        
        edge_dict[edge] = i
    
    sampled_edges = torch.tensor(np.random.choice(unique_edges, num_to_sample, replace=False))
    
    # print("hello")
    # print(len(sampled_edges))
    # for i in sampled_edges:
    #     print((data.edge_index[0][i], data.edge_index[1][i]))
    # print((data.edge_index[1][sampled_edges[i]], data.edge_index[0][sampled_edges[i]]) for i in range(len(sampled_edges)))
    

    reverse_edges = [edge_dict[(data.edge_index[1][i].item(), data.edge_index[0][i].item())] 
                     for i in sampled_edges]
    
    sampled_edges = torch.cat((sampled_edges, torch.tensor(reverse_edges)))    
    # get the unique endpoints of sampled edges as poisoned nodes in tensor

    sampled_nodes = torch.unique(data.edge_index[:, sampled_edges].flatten())
    # print("Sampling for Corrective")
    # print(sampled_nodes.shape)

    # print(len(sampled_edges))
    return sampled_edges, sampled_nodes

def get_closest_classes(classes, counts):
    '''
    returns the two classes with the closest number of samples in the training set
    '''

    pairwise_diffs = []
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            pairwise_diffs.append((classes[i], classes[j], abs(counts[i] - counts[j])))

    pairwise_diffs = sorted(pairwise_diffs, key=lambda x: x[2])

    return pairwise_diffs
