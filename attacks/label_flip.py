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