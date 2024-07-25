from torch_geometric import utils
import numpy as np
import torch
import random
import copy
from tqdm import trange, tqdm
import time
#graph is a standard geometric graph
from framework.trainer.base import NodeClassificationTrainer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
import torch.nn.functional as F

class EdgePoisonTrainer(NodeClassificationTrainer):
    def edge_attack_random_nodes(self, data, epsilon, seed):
        #returns a copy of the augmented edges and a list of nodes between which edges were added
        #adds edges between nodes having different classes
        np.random.seed(seed)
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
                    poisoned_nodes1.append(n1)
                    poisoned_nodes2.append(n2)
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

        return augmented_edge, torch.tensor(added_edge_indices, dtype=torch.long)
        #torch.tensor(poisoned_nodes1 , dtype=torch.long), torch.tensor(poisoned_nodes2 , dtype=torch.long)


    def edge_attack_specific_nodes(self, data, epsilon, seed, class1=0, class2=1):
        #returns a copy of the augmented edges and a list of nodes between which edges were added
        #adds edges between nodes having different classes
        seed=int(seed)
        np.random.seed(seed)

        #Sufficient classes do not exist
        num_classes= len(data.y.unique())
        if(class1>=num_classes or class2>=num_classes or class1==class2):
            raise KeyError("Invalid Classes")

        #Find nodes belonging to the chosen classes
        class1_indices= torch.where(data.y==class1)[0]
        class2_indices= torch.where(data.y==class2)[0]

        #Total possible edges fraction possible
        #Calculated by multiplying number of nodes belonging to each class- total possible edges
        if(epsilon<1):
            total_possible_edges= class1_indices.shape[0]*class2_indices.shape[0]
            epsilon= epsilon*total_possible_edges

        #Make sets to ensure edges already do not exist
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
        poisoned_edges= []

        #Add the required number of edges
        #Guaranteed different classes
        while count < epsilon:
            n1 = np.random.choice(class1_indices)
            n2 = np.random.choice(class2_indices)
            edge = (min(n1, n2), max(n1, n2))
            if edge not in existing_edges:
                to_arr.append(n1)
                from_arr.append(n2)
                existing_edges.add(edge)
                poisoned_nodes1.append(n1)
                poisoned_nodes2.append(n2)
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

        return augmented_edge, torch.tensor(added_edge_indices, dtype=torch.long)
        #torch.tensor(poisoned_nodes1 , dtype=torch.long), torch.tensor(poisoned_nodes2, dtype=torch.long)

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
            torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))

            print(f'Poisoning finished.')

def get_edge_poisoned_data(args, data, epsilon, seed):
    eg_trainer = EdgePoisonTrainer(args)
    aug, poisoned_indices = eg_trainer.edge_attack_specific_nodes(data, epsilon, seed)
    return aug, poisoned_indices