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
from scipy.spatial import ConvexHull

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

def rotate_embeddings(embeddings, angle):
    """
    Rotates the 2D embeddings by a specified angle.
    
    Args:
    embeddings (numpy.ndarray): The 2D embeddings to be rotated.
    angle (float): The angle (in degrees) by which to rotate the embeddings.
    
    Returns:
    numpy.ndarray: The rotated embeddings.
    """
    theta = np.radians(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                [np.sin(theta), np.cos(theta)]])
    
    return np.dot(embeddings, rotation_matrix)

def plot_embeddings(args, model, data, class1, class2, is_dr=False, mask="test", name=""):
    # Set the model to evaluation mode
    model.eval()

    # Forward pass: get embeddings
    with torch.no_grad():
        if is_dr and args.unlearning_model != "scrub":
            pre_embeddings = model(data.x, data.edge_index[:, data.dr_mask], get_pre_final=True)
        else:
            pre_embeddings = model(data.x, data.edge_index, get_pre_final=True)

    # If embeddings have more than 2 dimensions, apply t-SNE
    if pre_embeddings.shape[1] > 2:
        pre_embeddings = TSNE(n_components=2).fit_transform(pre_embeddings.cpu())
        pre_embeddings = torch.tensor(pre_embeddings).to(device)

    # Get the mask (either test, train, or val)
    if mask == "test":
        mask = data.test_mask
    elif mask == "train":
        mask = data.train_mask
    else:
        mask = data.val_mask

    # Filter embeddings and labels based on the mask
    pre_embeddings = pre_embeddings[mask]
    labels = data.y[mask]

    # Convert embeddings to numpy for processing
    pre_embeddings = pre_embeddings.cpu().numpy()
    pre_embeddings = rotate_embeddings(pre_embeddings, 270)
    # # Flip embeddings vertically by multiplying the y-dimension (2nd column) by -1
    # pre_embeddings[:, 0] = -pre_embeddings[:, 0]

    # Create masks for class1, class2, and other classes
    class1_mask = (labels == class1).cpu().numpy()
    class2_mask = (labels == class2).cpu().numpy()

    # Get unique class labels (excluding class1 and class2)
    unique_classes = torch.unique(labels).cpu().numpy()
    other_classes = [c for c in unique_classes if c != class1 and c != class2]

    plt.figure(figsize=(8, 8), tight_layout=True)
    sns.set(style="whitegrid")

    plt.grid(True, linestyle='-', alpha=0.7)

    # Convert labels to numpy for processing
    labels = labels.cpu().numpy()

    # Generate a color palette for all classes
    color_palette = sns.color_palette("pastel", len(unique_classes))

    # Plot each class with its unique color (non-poisoned classes will have reduced opacity)
    for i, cls in enumerate(other_classes):
        class_mask = (labels == cls)
        plt.scatter(
            pre_embeddings[class_mask, 0], 
            pre_embeddings[class_mask, 1], 
            color=color_palette[i], 
            alpha=0.25,  # Slightly lower opacity for non-poisoned classes
            s=50
        )

    # Plot class1 (poisoned class)
    plt.scatter(
        pre_embeddings[class1_mask, 0], 
        pre_embeddings[class1_mask, 1], 
        color='blue', 
        alpha=0.6, 
        s=120, 
        edgecolors='black', 
        linewidths=2  # Thicker borders
    )

    # Plot class2 (poisoned class)
    plt.scatter(
        pre_embeddings[class2_mask, 0], 
        pre_embeddings[class2_mask, 1], 
        color='red', 
        alpha=0.6, 
        s=120, 
        edgecolors='black', 
        linewidths=2  # Thicker borders
    )

    # Keep x-axis and y-axis ticks without labels and title
    plt.xticks(ticks=plt.xticks()[0], labels=[''] * len(plt.xticks()[0]))  # Remove x-axis labels
    plt.yticks(ticks=plt.yticks()[0], labels=[''] * len(plt.yticks()[0]))  # Remove y-axis labels
    plt.title('')  # Remove title
    plt.gca().legend().set_visible(False)  # Remove legend

    # Ensure the grid is visible
    plt.grid(True)

    # Save the plot
    os.makedirs("./plots", exist_ok=True)
    plt.savefig(f"./plots/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_{name}_embeddings_clean.png", format='png', bbox_inches='tight')
    plt.show()

def remove_outliers(data, threshold=1.0):
        # Calculate the convex hull of the data
        hull = ConvexHull(data)
        hull_points = data[hull.vertices]
        # Calculate the center of the hull
        center = np.mean(hull_points, axis=0)
        # Calculate distances from the center
        distances = np.linalg.norm(data - center, axis=1)
        # Calculate the threshold for outliers
        cutoff = np.percentile(distances, 100 * threshold)
        # Return data points that are within the threshold
        return data[distances <= cutoff]

def plot_poisoned_classes_only(args, model, data, class1, class2, is_dr=False, mask="test", name=""):
    # Set the model to evaluation mode
    model.eval()

    # Forward pass: get embeddings
    with torch.no_grad():
        if is_dr and args.unlearning_model != "scrub":
            pre_embeddings = model(data.x, data.edge_index[:, data.dr_mask], get_pre_final=True)
        else:
            pre_embeddings = model(data.x, data.edge_index, get_pre_final=True)

    # If embeddings have more than 2 dimensions, apply t-SNE
    if pre_embeddings.shape[1] > 2:
        pre_embeddings = TSNE(n_components=2).fit_transform(pre_embeddings.cpu())
        pre_embeddings = torch.tensor(pre_embeddings).to(device)

    # Get the mask (either test, train, or val)
    if mask == "test":
        mask = data.test_mask
    elif mask == "train":
        mask = data.train_mask
    else:
        mask = data.val_mask

    # Filter embeddings and labels based on the mask
    pre_embeddings = pre_embeddings[mask]
    labels = data.y[mask]

    # Convert embeddings and labels to numpy for processing
    pre_embeddings = pre_embeddings.cpu().numpy()
    pre_embeddings = rotate_embeddings(pre_embeddings, 75)
    labels = labels.cpu().numpy()

    # Create masks for class1 and class2 (poisoned classes)
    class1_mask = (labels == class1)
    class2_mask = (labels == class2)

    # Combine the embeddings for class1 and class2
    poisoned_embeddings = np.concatenate([pre_embeddings[class1_mask], pre_embeddings[class2_mask]])

    # shift_left_value = 50  # Adjust this value to control how far left the cluster moves
    print("=====================================")
    # print(poisoned_embeddings[:, 0])
    # poisoned_embeddings[:, 0] -= shift_left_value
    # poisoned_embeddings[:, 1] -= shift_left_value
    # print("=====================================")
    print(poisoned_embeddings[:, 1])

    # Prepare the plot
    plt.figure(figsize=(6, 6))
    sns.set(style="whitegrid")

    # Plot class1 (poisoned class) embeddings
    plt.scatter(
        poisoned_embeddings[:sum(class1_mask), 0], 
        poisoned_embeddings[:sum(class1_mask), 1], 
        color='blue', 
        alpha=0.7, 
        s=120, 
        edgecolors='black', 
        linewidths=2,  # Thicker borders
        label=f'Class {class1}'
    )

    # Plot class2 (poisoned class) embeddings
    plt.scatter(
        poisoned_embeddings[sum(class1_mask):, 0], 
        poisoned_embeddings[sum(class1_mask):, 1], 
        color='red', 
        alpha=0.7, 
        s=120, 
        edgecolors='black', 
        linewidths=2,  # Thicker borders
        label=f'Class {class2}'
    )
    # print("=====================================")
    # print(poisoned_embeddings[:, 1])

    # plt.xlim(50, 100)
    plt.ylim(50, 110)

    # Set the plot properties: no axis labels, grid, etc.
    plt.xticks([])  # Remove x-axis labels
    plt.yticks([])  # Remove y-axis labels
    plt.title('')  # Remove title
    plt.grid(True)

    # Add a legend for the two poisoned classes
    plt.gca().legend().set_visible(False)  # Remove legend

    # Save the plot as PNG
    os.makedirs("./plots", exist_ok=True)
    plt.savefig(f"./plots/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_{name}_poisoned_classes_zoom.png", format='png', bbox_inches='tight')
    plt.show()

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
