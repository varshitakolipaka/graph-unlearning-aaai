import torch
import numpy as np
import random

import torch
import numpy as np
import random
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import copy

def apply_poison_clean(data, samples, trigger_size, poisoned_feature_indices=None):
    poisoned_data = data.clone()
    if poisoned_feature_indices is None:
        poisoned_feature_indices = [i for i in range(trigger_size) ]

    # Ensure samples are in a list for consistent iteration
    if isinstance(samples, torch.Tensor):
        samples = samples.tolist()

    # Apply the trigger and set labels to target_class
    for node in samples:
        # Apply the trigger by setting selected features to a high value (e.g., 12345)
        poisoned_data.x[node, poisoned_feature_indices] = 1  # Adjust this value based on feature scaling
        print(sum(poisoned_data.x[node][:trigger_size]))

    return poisoned_data, poisoned_feature_indices


def clean_label(data, epsilon, seed, victim_class=68, trigger_size=15, test_poison_fraction=1):
    data = data.cpu()
    data = copy.deepcopy(data)
    data.victim_class = victim_class

    train_mask = data.train_mask
    # Get all victim class nodes that are in the train mask
    class_indices = [i.item() for i in (data.y == victim_class).nonzero(as_tuple=True)[0] if train_mask[i]]
    num_class_nodes = len(class_indices)
    print(num_class_nodes)

    if num_class_nodes == 0:
        raise ValueError(f"No nodes found for victim class {victim_class}.")

    if epsilon <= 1:
        # Calculate number of nodes to poison based on epsilon (fraction of victim class nodes)
        num_poison = int(epsilon * num_class_nodes)
        print(f"Poisoning {num_poison} out of {num_class_nodes} victim class nodes.")
    else:
        # If epsilon >1, treat it as the exact number of nodes to poison
        num_poison = min(int(epsilon), len(class_indices))
        print(f"Poisoning {num_poison} victim class nodes.")

    # Ensure that we don't poison more nodes than available in the class
    num_poison = min(num_poison, num_class_nodes)

    # Get train mask and filter only nodes in the victim class and train mask
    train_mask = data.train_mask
    train_class_indices = [i for i in class_indices if train_mask[i]]

    if len(train_class_indices) == 0:
        raise ValueError("No training nodes found in the victim class.")

    # Randomly select nodes to poison from the victim class in the train mask
    if num_poison < len(train_class_indices):
        poisoned_nodes = random.sample(train_class_indices, num_poison)
    else:
        poisoned_nodes = train_class_indices


    poisoned_indices = torch.tensor(poisoned_nodes, dtype=torch.long)


    # Create and apply the trigger to the selected nodes
    poisoned_data, poisoned_feature_indices = apply_poison_clean(
        data,
        poisoned_indices,
        trigger_size,
    )
    poisoned_data.poisoned_feature_indices = poisoned_feature_indices

    # Now handle test set poisoning (victim class only)
    test_mask = poisoned_data.test_mask
    victim_test_indices = [
        i for i in range(poisoned_data.num_nodes)
        if test_mask[i] and poisoned_data.y[i] == victim_class  # Poison only victim class nodes in test
    ]

    # Apply the poison to a fraction of victim class test nodes
    num_victim_test_nodes = len(victim_test_indices)
    print("meow check", num_victim_test_nodes)
    if num_victim_test_nodes > 0:
        num_test_poison = num_victim_test_nodes
        poisoned_test_nodes = random.sample(victim_test_indices, num_test_poison)
        print(f"Poisoning {len(poisoned_test_nodes)} test nodes out of {num_victim_test_nodes} victim class nodes.")

        # Create a binary mask for poisoned test nodes
        poison_test_mask = torch.zeros(poisoned_data.num_nodes, dtype=torch.bool)
        poison_test_mask[poisoned_test_nodes] = True

        # Apply the same trigger to the poisoned test nodes
        poisoned_data, _ = apply_poison_clean(
            poisoned_data,
            poisoned_test_nodes,
            trigger_size,
            poisoned_feature_indices=poisoned_feature_indices
        )
        poisoned_data.poison_test_mask = poison_test_mask
    else:
        print("No victim class nodes found in the test set.")
        poisoned_data.poison_test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    poisoned_data.poisoned_nodes = poisoned_indices
    return poisoned_data, poisoned_indices