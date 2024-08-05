import torch
import numpy as np
import random
import copy
from torch_geometric import utils
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def edge_attack_random_nodes(data, epsilon, seed):
    np.random.seed(seed)
    random.seed(seed)
    data= data.cpu()
    if(epsilon<1):
        epsilon= epsilon*data.num_edges

    N = data.num_nodes
    to_arr = []
    from_arr = []

    existing_edges = set((min(n1, n2), max(n1, n2)) for n1, n2 in data.edge_index.t().tolist())
    poisoned_edges = set()
    poisoned_nodes= set()


    while len(poisoned_edges) < epsilon:
        n1 = random.randint(0, N-1)
        n2 = random.randint(0, N-1)
        if n1 != n2 and data.y[n1] != data.y[n2]:
            edge = (min(n1, n2), max(n1, n2))
            if edge not in existing_edges and edge not in poisoned_edges:
                to_arr.append(n1)
                from_arr.append(n2)
                poisoned_edges.add(edge)
                poisoned_nodes.add(n1)
                poisoned_nodes.add(n2)

    to_arr = torch.tensor(to_arr, dtype=torch.int64)
    from_arr = torch.tensor(from_arr, dtype=torch.int64)
    edge_index_to_add = torch.vstack([to_arr, from_arr])
    edge_index_to_add = utils.to_undirected(edge_index_to_add)

    data.poisoned_nodes= torch.tensor(list(poisoned_nodes), dtype=torch.long)
    augmented_edge = torch.cat([data.edge_index, edge_index_to_add], dim=1)
    data.edge_index = augmented_edge
    nums= list(range(len(data.edge_index[0])-2*len(poisoned_edges), len(data.edge_index[0])))
    return data, torch.tensor(nums, dtype=torch.long)


def edge_attack_specific_nodes(data, epsilon, seed, class1=None, class2=None):
    np.random.seed(seed)
    random.seed(seed)
    data.og_edges= data.edge_index.clone()

    # Get train indices and class counts
    train_indices = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_labels = data.y[train_indices]

    if class1 is None or class2 is None:
        counts = torch.bincount(train_labels)
        class1, class2 = torch.topk(counts, 2).indices

    class1_indices = torch.where(data.y == class1)[0]
    class2_indices = torch.where(data.y == class2)[0]

    class1_indices=class1_indices.cpu().detach().numpy()
    class2_indices=class2_indices.cpu().detach().numpy()

    data.class1 = class1
    data.class2 = class2

    # Determine number of edges to add
    if(epsilon<1):
        total_possible_edges = class1_indices.size(0) * class2_indices.size(0)
        epsilon = min(int(epsilon * total_possible_edges), total_possible_edges)

    # Set of existing edges
    existing_edges = set((min(n1, n2), max(n1, n2)) for n1, n2 in data.edge_index.t().tolist())

    to_arr = []
    from_arr = []
    poisoned_nodes= set()
    poisoned_edges = set()

    # Use tqdm to show progress
    while len(poisoned_edges) < epsilon:
        n1 = np.random.choice(class1_indices)
        n2 = np.random.choice(class2_indices)
        edge = (min(n1, n2), max(n1, n2))

        if edge not in existing_edges and edge not in poisoned_edges:
            to_arr.append(n1)
            from_arr.append(n2)
            poisoned_edges.add(edge)
            poisoned_nodes.add(n1)
            poisoned_nodes.add(n2)

    to_arr = torch.tensor(to_arr, dtype=torch.int64)
    from_arr = torch.tensor(from_arr, dtype=torch.int64)
    edge_index_to_add = torch.stack([to_arr, from_arr], dim=0)
    edge_index_to_add = utils.to_undirected(edge_index_to_add)
    edge_index_to_add = edge_index_to_add.clone().detach().to(device)

    data.poisoned_nodes= torch.tensor(list(poisoned_nodes), dtype=torch.long)
    augmented_edge = torch.cat([data.edge_index, edge_index_to_add], dim=1)
    data.edge_index = augmented_edge
    nums= list(range(len(data.edge_index[0])-2*len(poisoned_edges), len(data.edge_index[0])))
    return data, torch.tensor(nums, dtype=torch.long)
