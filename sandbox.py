from framework.utils import get_original_data, train_test_split
from attacks.label_flip import label_flip_attack
from torch_geometric.utils import degree
import numpy as np
import torch

seed=0
class1=0
class2=2
epsilon=0.5

data= get_original_data("Cora")
data, _, _ = train_test_split(data, 0, train_ratio=0.7, val_ratio=0.1)

train_indices = data.train_mask.nonzero(as_tuple=False).view(-1)
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


exit(0)
poisoned_nodes, poisoned_indices= label_flip_attack(data, df_size, seed, 0, 2)