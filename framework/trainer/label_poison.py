
import os
import time
import numpy as np
from framework.utils import EarlyStopping, wandb_log
import torch
from tqdm import tqdm, trange
import torch.nn as nn
import torch.nn.functional as F

from framework.trainer.base import NodeClassificationTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LabelPoisonTrainer(NodeClassificationTrainer):
    def attack(self, data, epsilon, seed):
        np.random.seed(seed)
        
        if epsilon > 1:
            epsilon = epsilon / 100 # Convert percentage to fraction
    
        # Count the number of nodes for each class in the training set
        train_indices = data.train_mask.nonzero(as_tuple=False).view(-1)
        train_labels, counts = torch.unique(data.y[train_indices], return_counts=True)
        
        # Get the two most frequent classes
        sorted_indices = torch.argsort(counts, descending=True)
        class1, class2 = train_labels[sorted_indices[:2]]
        
        # Find indices of these classes in the training set
        class1_indices = train_indices[data.y[train_indices] == class1]
        class2_indices = train_indices[data.y[train_indices] == class2]
        
        print(f'Class {class1} has {len(class1_indices)} nodes')
        print(f'Class {class2} has {len(class2_indices)} nodes')
        
        # Determine the number of flips
        num_flips = int(epsilon * len(train_indices))
        
        # Randomly select indices to flip
        flip_indices_class1 = np.random.choice(class1_indices, num_flips // 2, replace=False)
        flip_indices_class2 = np.random.choice(class2_indices, num_flips // 2, replace=False)
        
        # Perform the label flip
        data.y[flip_indices_class1] = class2
        data.y[flip_indices_class2] = class1
        
        data.poison_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.poison_mask[flip_indices_class1] = 1
        data.poison_mask[flip_indices_class2] = 1
        
        return data
    
    def train(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        model = model.to(device)
        data = data.to(device)
        
        for epoch in trange(args.epochs, desc="Poisoning"):
            model.train()
            
            start_time = time.time()
            total_step = 0
            total_loss = 0
            
            z = model(data.x, data.edge_index)
            loss = F.nll_loss(z[data.train_mask], data.y[data.train_mask])

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            total_step += 1
            total_loss += loss.item()

            end_time = time.time()
            epoch_time = end_time - start_time
            
            step_log = {
                'Epoch': epoch,
                'train_loss': loss.item(),
                'train_time': epoch_time
            }
            print(step_log)
            # wandb_log(step_log)
            msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in step_log.items()]
            
            # Save
            ckpt = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }
            torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))

            print(f'Poisoning finished.')
            
def get_label_poisoned_data(data, epsilon, seed):
    lb_trainer = LabelPoisonTrainer()
    data = lb_trainer.attack(data, epsilon, seed)
    