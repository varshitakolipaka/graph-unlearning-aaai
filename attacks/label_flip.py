import torch
import numpy as np
from framework.utils import get_closest_classes
from torch_geometric.utils import degree

def label_flip_attack(data, epsilon, seed, class1=None, class2=None):
    np.random.seed(seed)
    data= data.cpu()
    train_indices = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_labels, counts = torch.unique(data.y[train_indices], return_counts=True)

    for i, label in enumerate(train_labels):
        print(f"Class {label}: {counts[i]}")

    if class1 is None or class2 is None:
        class_pairs = get_closest_classes(train_labels, counts)

        class1, class2, _ = class_pairs[0]


    class1_indices = train_indices[data.y[train_indices] == class1]
    class2_indices = train_indices[data.y[train_indices] == class2]

    # epsilon is the fraction of class indices to flip, at max half of the class indices
    epsilon = min(epsilon, 0.5)
    num_flips = int(epsilon * min(len(class1_indices), len(class2_indices)))

    print(f"Flipping {num_flips} labels from class {class1} to class {class2} and vice versa")

    # Randomly select indices to flip
    flip_indices_class1 = np.random.choice(class1_indices, num_flips, replace=False)
    flip_indices_class2 = np.random.choice(class2_indices, num_flips, replace=False)
    data.y[flip_indices_class1] = class2
    data.y[flip_indices_class2] = class1

    data.class1 = class1
    data.class2 = class2

    print(f"Poisoned {num_flips} labels in total from class {class1} and class {class2}")

    flipped_indices = np.concatenate([flip_indices_class1, flip_indices_class2])
    data.poisoned_nodes = torch.tensor(flipped_indices)

    data.poison_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.poison_mask[list(flipped_indices)] = True

    return data, flipped_indices

def label_flip_attack_strong(data, epsilon, seed, class1=None, class2=None):
    np.random.seed(seed)
    data= data.cpu()
    train_indices = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_labels, counts = torch.unique(data.y[train_indices], return_counts=True)

    if class1 is None or class2 is None:
        class_pairs = get_closest_classes(train_labels, counts)
        class1, class2, _ = class_pairs[0]

    class1_indices = train_indices[data.y[train_indices] == class1]
    class2_indices = train_indices[data.y[train_indices] == class2]

    deg= degree(data.edge_index[0])
    sorted_indices = torch.argsort(deg, descending=True)
    index_map = {idx.item(): pos for pos, idx in enumerate(sorted_indices)}

    sorted_class1_indices = [i.item() for i in sorted(class1_indices, key=lambda x: index_map[x.item()])]
    sorted_class2_indices = [i.item() for i in sorted(class2_indices, key=lambda x: index_map[x.item()])]

    epsilon = min(epsilon, 0.5)
    num_flips = int(epsilon * min(len(class1_indices), len(class2_indices)))
    flip_indices_class1 = sorted_class1_indices[:num_flips]
    flip_indices_class2 = sorted_class2_indices[:num_flips]

    # # sort the nodes by degree
    # degrees = data.adj.sum(1)
    # degrees = degrees.squeeze()
    # sorted_indices = np.argsort(degrees)
    # class1_indices = np.intersect1d(class1_indices, sorted_indices)
    # class2_indices = np.intersect1d(class2_indices, sorted_indices)

    # # epsilon is the fraction of class indices to flip, at max half of the class indices
    # epsilon = min(epsilon, 0.5)
    # num_flips = int(epsilon * min(len(class1_indices), len(class2_indices)))

    # print(f"Flipping {num_flips} labels from class {class1} to class {class2} and vice versa")

    # # Randomly select indices to flip
    # # flip_indices_class1 = np.random.choice(class1_indices, num_flips, replace=False)
    # # flip_indices_class2 = np.random.choice(class2_indices, num_flips, replace=False)

    data.y[flip_indices_class1] = class2
    data.y[flip_indices_class2] = class1

    data.class1 = class1
    data.class2 = class2

    print(f"Poisoned {num_flips} labels in total from class {class1} and class {class2}")

    flipped_indices = np.concatenate([flip_indices_class1, flip_indices_class2])
    data.poisoned_nodes = torch.tensor(flipped_indices)

    data.poison_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.poison_mask[list(flipped_indices)] = True

    return data, flipped_indices