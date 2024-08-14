import torch
import numpy as np
import random
from framework.utils import seed_everything

def generate_trigger(size, num_ones, seed):
    seed_everything(seed)
    tensor = torch.cat([torch.zeros(size-num_ones), torch.ones(num_ones)])
    return tensor[torch.randperm(size)]

def poison_train_mask(data, epsilon, seed, target_class=None):
    if(target_class==None):
        target_class=0

    seed_everything(seed)
    data= data.cpu()

    data.trigger= generate_trigger(len(data.x[0]), int(len(data.x[0])/5), seed)
    data.target_class= target_class
    train_indices= data.train_mask.nonzero(as_tuple=False).view(-1)
    if(epsilon<1):
        epsilon= int(epsilon*len(train_indices))
    poisonable_indices= torch.tensor([i.item() for i in train_indices if data.y[i]!=data.target_class], dtype=torch.long)
    idx = torch.randperm(poisonable_indices.size(0))[:epsilon]
    samples = poisonable_indices[idx]

    data.x[samples]=data.trigger
    data.y[samples]=data.target_class
    return data, samples

def poison_test_mask(data, epsilon, seed):
    seed_everything(seed)
    data= data.cpu()

    test_indices= data.test_mask.nonzero(as_tuple=False).view(-1)
    if(epsilon<1):
        epsilon= int(epsilon*len(test_indices))
    poisonable_indices= torch.tensor([i.item() for i in test_indices if data.y[i]!=data.target_class], dtype=torch.long)
    idx = torch.randperm(poisonable_indices.size(0))[:epsilon]
    samples = poisonable_indices[idx]

    data.x[samples]=data.trigger
    data.y[samples]=data.target_class

    data.test_mask[samples]=False
    data.poison_test_mask = torch.zeros(len(data.test_mask), dtype=torch.bool)
    data.poison_test_mask[samples]=True

def trigger_attack(data, epsilon, seed, test_poison_fraction, target_class=None):
    poisoned_data, poisoned_indicies= poison_train_mask(data, epsilon, seed, target_class)
    print(f"Trigger: {poisoned_data.trigger}")
    poison_test_mask(data, test_poison_fraction, seed)
    return poisoned_data, poisoned_indicies