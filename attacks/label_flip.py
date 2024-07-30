import torch
import numpy as np

def label_flip_attack(data, epsilon, seed):
    np.random.seed(seed)
    data= data.cpu()
    train_indices = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_labels, counts = torch.unique(data.y[train_indices], return_counts=True)
    sorted_indices = torch.argsort(counts, descending=True)
    class1, class2 = train_labels[sorted_indices[:2]]
    class1_indices = train_indices[data.y[train_indices] == class1]
    class2_indices = train_indices[data.y[train_indices] == class2]
    # Determine the number of flips
    num_flips = int(epsilon * len(train_indices))

    # Randomly select indices to flip
    flip_indices_class1 = np.random.choice(class1_indices, num_flips // 2, replace=False)
    flip_indices_class2 = np.random.choice(class2_indices, num_flips // 2, replace=False)
    data.y[flip_indices_class1] = class2
    data.y[flip_indices_class2] = class1
    flipped_indices = np.concatenate([flip_indices_class1, flip_indices_class2])
    return data, flipped_indices