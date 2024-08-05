import torch
import numpy as np
import random
import copy
from torch_geometric import utils

def edge_attack_random_nodes(data, epsilon, seed):
    np.random.seed(seed)
    random.seed(seed)
    data= data.cpu()
    if(epsilon<1):
        epsilon= epsilon*data.num_edges

    N = data.num_nodes
    existing_edges = set()
    poisoned_nodes1= []
    poisoned_nodes2= []
    for i in range(data.edge_index.size(1)):
        n1, n2 = data.edge_index[:, i].tolist()
        existing_edges.add((min(n1, n2), max(n1, n2)))

    to_arr = []
    from_arr = []
    count = 0
    poisoned_edges = []

    while count < epsilon:
        n1 = random.randint(0, N-1)
        n2 = random.randint(0, N-1)
        if n1 != n2 and data.y[n1] != data.y[n2]:
            edge = (min(n1, n2), max(n1, n2))
            if edge not in existing_edges:
                to_arr.append(n1)
                from_arr.append(n2)
                existing_edges.add(edge)

                #forward add
                poisoned_nodes1.append(n1)
                poisoned_nodes2.append(n2)

                #backward add
                poisoned_nodes1.append(n2)
                poisoned_nodes2.append(n1)

                poisoned_edges.append(edge)
                count += 1

    to_arr = torch.tensor(to_arr, dtype=torch.int64)
    from_arr = torch.tensor(from_arr, dtype=torch.int64)
    edge_index_to_add = torch.vstack([to_arr, from_arr])
    edge_index_to_add = utils.to_undirected(edge_index_to_add)
    edge_copy= copy.deepcopy(data.edge_index)
    augmented_edge= torch.cat([edge_copy, edge_index_to_add], dim=1)
    augmented_edge= utils.sort_edge_index(augmented_edge)

    added_edge_indices = []
    for i in range(augmented_edge.size(1)):
        edge = (min(augmented_edge[0, i].item(), augmented_edge[1, i].item()), max(augmented_edge[0, i].item(), augmented_edge[1, i].item()))
        if edge in poisoned_edges:
            added_edge_indices.append(i)
    data.edge_index= augmented_edge
    return data, torch.tensor(added_edge_indices, dtype=torch.long)
    # torch.tensor(list(set(poisoned_nodes1)) , dtype=torch.long)

from tqdm import tqdm  # Import tqdm

def edge_attack_specific_nodes(data, epsilon, seed):
    np.random.seed(seed)
    random.seed(seed)
    data.og_edges= data.edge_index.clone()

    # Get train indices and class counts
    train_indices = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_labels = data.y[train_indices]
    counts = torch.bincount(train_labels)
    class1, class2 = torch.topk(counts, 2).indices

    class1_indices = torch.where(data.y == class1)[0]
    class2_indices = torch.where(data.y == class2)[0]

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
        n1 = np.random.choice(class1_indices.numpy())
        n2 = np.random.choice(class2_indices.numpy())
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

    data.poisoned_nodes= torch.tensor(list(poisoned_nodes), dtype=torch.long)
    augmented_edge = torch.cat([data.edge_index, edge_index_to_add], dim=1)
    data.edge_index = augmented_edge
    nums= list(range(len(data.edge_index[0])-2*len(poisoned_edges), len(data.edge_index[0])))
    return data, torch.tensor(nums, dtype=torch.long)
