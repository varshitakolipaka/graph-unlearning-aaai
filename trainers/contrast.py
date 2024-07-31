import os, math
import copy
import time
#import wandb
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling, k_hop_subgraph
from torch_geometric.loader import GraphSAINTRandomWalkSampler

from .base import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ContrastiveUnlearnTrainer(Trainer):
    def __init__(self, model, data, optimizer, args):
        super().__init__(model, data, optimizer)
        self.args= args
        self.attacked_idx = data.attacked_idx

    def task_loss(self, model, data):
        out = model(data.x, data.edge_index)
        criterion = torch.nn.CrossEntropyLoss()
        # use the retain mask to calculate the loss
        try:
            mask = data.retain_mask
        except:
            mask = data.train_mask
        loss = criterion(out[mask], data.y[mask])
        return loss


    def contrastive_loss(self, pos_dist, neg_dist, margin, mask):
        # take the retained mask into account
        mask_cpu = mask.cpu()
    
        pos_dist = pos_dist[mask_cpu]
        neg_dist = neg_dist[mask_cpu]

        pos_loss = torch.mean(pos_dist**2)
        neg_loss = torch.mean(F.relu(margin - neg_dist) ** 2)
        loss = pos_loss + neg_loss
        return loss


    def unlearn_loss(
        self, model, data, pos_dist, neg_dist, margin=1.0, lmda = 0.8
    ):
        try:
            mask = data.retain_mask
        except:
            mask = data.train_mask
        return lmda * self.task_loss(model, data) + (1 - lmda) * self.contrastive_loss(
            pos_dist, neg_dist, margin, mask
        )


    def calc_distance(self, embeddings, node, positive_samples, negative_samples):
        # Contrastive loss


        anchor = embeddings[node].unsqueeze(0)
        positive = embeddings[positive_samples]
        negative = embeddings[negative_samples]

        # cosine similarity
        # pos_dist = 1 - F.cosine_similarity(anchor, positive).mean()
        # neg_dist = 1 - F.cosine_similarity(anchor, negative).mean()

        pos_dist = (
            (anchor.unsqueeze(1) - positive).pow(2).sum(-1).mean(1)
        )  # Euclidean distance between anchor and positive
        neg_dist = (
            (anchor.unsqueeze(1) - negative).pow(2).sum(-1).mean(1)
        )  # Euclidean distance between anchor and negative

        return pos_dist, neg_dist

    def store_subset(self, data, idx):
        # store the subset of the idx in a dictionary
        subset_dict = {}
        for idx in range(len(data.train_mask)):
            if data.retain_mask[idx]:
                subset, edge_index, _, _ = k_hop_subgraph(idx, 2, data.edge_index)
                subset_set = set(subset.tolist())
                subset_dict[idx] = subset_set
        self.subset_dict = subset_dict
    
    def store_edge_index_for_poison(self, data, idx):
        edge_index_for_poison_dict = {}
        for idx in range(len(data.train_mask)):
            if data.retain_mask[idx]:
                _, edge_index_for_poison, _, _ = k_hop_subgraph(
                    idx, 1, data.edge_index
                )
                edge_index_for_poison_dict[idx] = edge_index_for_poison
        self.edge_index_for_poison_dict = edge_index_for_poison_dict

    def get_distances(self, model, data, attacked_set):
        # get pos, neg distances for nodes beforehand
        pos_dist = []
        neg_dist = []
        
        embeddings = model.conv1(data.x, data.edge_index)
        embeddings = F.relu(embeddings)
        embeddings = F.dropout(embeddings, training=model.training)
        embeddings = model.conv2(embeddings, data.edge_index)
        
        for idx in range(len(data.train_mask)):
            if data.retain_mask[idx]:
                # Get K-hop subgraph
                subset_set = self.subset_dict[idx]

                # Define positive and negative samples
                positive_samples = subset_set - attacked_set
                negative_samples = attacked_set

                if not positive_samples or not negative_samples:
                    pos_dist.append(0)
                    neg_dist.append(0)
                    continue

                positive_samples = list(positive_samples)
                negative_samples = list(negative_samples)

                # Compute distances
                pos, neg = self.calc_distance(embeddings, idx, positive_samples, negative_samples)
                pos_dist.append(pos.item())
                neg_dist.append(neg.item())
            else:
                pos_dist.append(0)
                neg_dist.append(0)

        # convert to tensor
        pos_dist = torch.tensor(pos_dist)
        neg_dist = torch.tensor(neg_dist)
        return pos_dist, neg_dist

    def get_distances_edge(self, model, data, attacked_edge_list):
        # attacked edge index contains all the edges that were maliciously added

        # get pos, neg distances for nodes beforehand
        pos_dist = []
        neg_dist = []
        
        embeddings = model.conv1(data.x, data.edge_index)
        embeddings = F.relu(embeddings)
        embeddings = F.dropout(embeddings, training=model.training)
        embeddings = model.conv2(embeddings, data.edge_index)
        
        for idx in range(len(data.train_mask)):
            if data.retain_mask[idx]:
                # Get K-hop subgraph
                subset_set = self.subset_dict[idx]
                edge_index_for_poison = self.edge_index_for_poison_dict[idx]

                # convert edge_index to a list of tuples
                edge_index_list = [
                    (edge_index_for_poison[0][i].item(), edge_index_for_poison[1][i].item())
                    for i in range(edge_index_for_poison.shape[1])
                ]
                # check for intersection between the edge_index_list and attacked_edge_list
                attacked_edge_set = set(attacked_edge_list)

                intersection = set(edge_index_list).intersection(attacked_edge_set)

                if intersection:
                    attacked_set = set()
                    for edge in intersection:
                        # add the node which is not the idx
                        u, v = edge
                        if u == idx:
                            attacked_set.add(v)
                        else:
                            attacked_set.add(u)

                    positive_samples = subset_set - attacked_set
                    negative_samples = attacked_set
                else:
                    pos_dist.append(0)
                    neg_dist.append(0)
                    continue

                if not positive_samples or not negative_samples:
                    pos_dist.append(0)
                    neg_dist.append(0)
                    continue

                positive_samples = list(positive_samples)
                negative_samples = list(negative_samples)

                # Compute distances
                pos, neg = self.calc_distance(embeddings, idx, positive_samples, negative_samples)
                pos_dist.append(pos.item())
                neg_dist.append(neg.item())

        # convert to tensor
        pos_dist = torch.tensor(pos_dist)
        neg_dist = torch.tensor(neg_dist)
        return pos_dist, neg_dist

    def train_node(self):
        
        # attacked idx must be a list of nodes
        model = self.model
        data = self.data
        args = self.args
        attacked_idx = self.attacked_idx
        optimizer = self.optimizer
        
        model = model.to(device)
        data = data.to(device)
        
        attacked_set = set(attacked_idx.tolist())
        
        for epoch in trange(args.contrastive_epochs_1, desc="Unlearning 1"):
            pos_dist, neg_dist = self.get_distances(model, data, attacked_set)
            loss = self.unlearn_loss(
                model,
                data,
                pos_dist,
                neg_dist,
                margin=args.contrastive_margin,
                lmda = args.contrastive_lambda
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        for epoch in trange(args.contrastive_epochs_2, desc="Unlearning 2"):
            pos_dist, neg_dist = self.get_distances(model, data, attacked_set)
            loss = self.unlearn_loss(
                model,
                data,
                pos_dist,
                neg_dist,
                margin=args.contrastive_margin,
                lmda = 1
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    
    def train_edge(self):
        # attack idx must be a list of tuples (u,v)
        model = self.model
        data = self.data
        args = self.args
        attacked_idx = self.attacked_idx
        optimizer = self.optimizer
        
        for epoch in trange(args.contrastive_epochs_1, desc="Unlearning 1"):
            pos_dist, neg_dist = self.get_distances_edge(
                model, data, attacked_idx
            )
            loss = self.unlearn_loss(
                model,
                data,
                pos_dist,
                neg_dist,
                margin=args.contrastive_margin,
                lmda= args.contrastive_lambda
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        for epoch in trange(args.contrastive_epochs_2, desc="Unlearning 2"):
            pos_dist, neg_dist = self.get_distances(model, data, attacked_idx)
            loss = self.unlearn_loss(
                model,
                data,
                pos_dist,
                neg_dist,
                margin=args.contrastive_margin,
                lmda = 1
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def train(self):
        
        # attack_idx is an extra needed parameter which is defined above in both node and edge functions
        self.data.retain_mask = self.data.train_mask.clone()
        self.store_subset(self.data, self.attacked_idx)
        start_time = time.time()
        if self.args.request == "node":
            self.train_node()
        elif self.args.request == "edge":
            self.store_edge_index_for_poison(self.data, self.attacked_idx)
            self.train_edge()
        end_time = time.time()
        train_acc, msc_rate, f1 = self.evaluate(is_dr=True)
        print(f'Train Acc: {train_acc}, Misclassification: {msc_rate},  F1 Score: {f1}')
        
        print(f"Training time: {end_time - start_time}")
        
        return train_acc, msc_rate, end_time - start_time