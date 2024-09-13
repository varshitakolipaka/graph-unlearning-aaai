import os, math
import copy
from pprint import pprint
import time
import scipy.sparse as sp
# import wandb
import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling, k_hop_subgraph, to_scipy_sparse_matrix
from torch_geometric.loader import GraphSAINTRandomWalkSampler

from .base import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        #print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds to execute.")
        return result

    return wrapper


class ContrastiveUnlearnTrainer(Trainer):
    def __init__(self, model, data, optimizer, args):
        super().__init__(model, data, optimizer)
        self.args = args
        self.attacked_idx = data.attacked_idx
        self.embeddings = None
        self.criterion = torch.nn.CrossEntropyLoss()

    def reverse_features(self, features):
        reverse_features = features.clone()
        if self.args.request== "edge":
            for idx in self.data.poisoned_nodes:
                reverse_features[idx] = 1-reverse_features[idx]
            return reverse_features

        for idx in self.attacked_idx:
            reverse_features[idx] = 1-reverse_features[idx]
        return reverse_features

    def get_sample_points(self):
        if self.args.request == "edge":
            og_logits = F.softmax(self.model(self.data.x, self.data.edge_index), dim=1)
            temp_features = self.data.x.clone()
            reverse_feature = self.reverse_features(temp_features)
            final_logits = F.softmax(self.model(reverse_feature, self.data.edge_index), dim=1)
            diff = torch.abs(og_logits - final_logits)
            diff = torch.mean(diff, dim=1)
            diff = diff[self.data.poisoned_nodes]
            frac = self.args.contrastive_frac
            _, indices = torch.topk(diff, int(frac * len(self.data.poisoned_nodes)), largest=True)
            influence_nodes_with_unlearning_nodes = self.data.poisoned_nodes[indices]
            print(f"Nodes influenced: {len(influence_nodes_with_unlearning_nodes)}")
            self.data.sample_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
            self.data.sample_mask[influence_nodes_with_unlearning_nodes] = True

            poisoned_edges = self.data.edge_index[:, self.attacked_idx]
            negative_sample_dict= {int: set()}

            for i in range(len(poisoned_edges[0])):
                toNode= poisoned_edges[0][i].item()
                fromNode= poisoned_edges[1][i].item()

                if toNode not in negative_sample_dict:
                    negative_sample_dict[toNode] = set()
                negative_sample_dict[toNode].add(fromNode)

                if fromNode not in negative_sample_dict:
                    negative_sample_dict[fromNode] = set()
                negative_sample_dict[fromNode].add(toNode)
            self.negative_sample_dict= negative_sample_dict
            return

        subset, _, _, _ = k_hop_subgraph(
            self.attacked_idx.clone().detach(), self.args.k_hop, self.data.edge_index
        )

        # remove attacked nodes from the subset
        subset = subset[~np.isin(subset.cpu(), self.attacked_idx.cpu())]

        og_logits = F.softmax(self.model(self.data.x, self.data.edge_index), dim=1)
        temp_features = self.data.x.clone()
        reverse_feature = self.reverse_features(temp_features)
        final_logits = F.softmax(self.model(reverse_feature, self.data.edge_index), dim=1)

        diff = torch.abs(og_logits - final_logits)

        # average across all classes
        diff = torch.mean(diff, dim=1)

        # take diffs of only the subset without the attacked nodes
        diff = diff[subset]

        #  get the top 10% of the indices
        frac = self.args.contrastive_frac
        _, indices = torch.topk(diff, int(frac * len(subset)), largest=True)

        influence_nodes_with_unlearning_nodes = indices

        print(f"Nodes influenced: {len(influence_nodes_with_unlearning_nodes)}")

        self.data.sample_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        self.data.sample_mask[influence_nodes_with_unlearning_nodes] = True

    @time_it
    def task_loss(self, mask=None, ascent=False):        
        # use the retain mask to calculate the loss
        embeddings = self.model(self.data.x, self.data.edge_index[:, self.data.dr_mask])
        
        if mask is None:    
            try:
                mask = self.data.retain_mask
            except:
                mask = self.data.train_mask
        
        if ascent:
            mask = self.data.poison_mask

        loss = self.criterion(embeddings[mask], self.data.y[mask])
        
        if ascent:
            return -loss
        
        return loss

    @time_it
    def contrastive_loss(self, pos_dist, neg_dist, margin):
        pos_loss = torch.mean(pos_dist)
        neg_loss = torch.mean(F.relu(margin - neg_dist))
        loss = pos_loss + neg_loss
        return loss

    @time_it
    def unlearn_loss(self, pos_dist, neg_dist, margin=1.0, lmda=0.8, ascent=False):
        if lmda == 1:
            return self.task_loss()
        return lmda * self.task_loss(ascent=ascent) + (1 - lmda) * self.contrastive_loss(
            pos_dist, neg_dist, margin
        )

    def calc_distances(self, nodes, positive_samples, negative_samples):
        # Vectorized contrastive loss calculation
        anchors = self.embeddings[nodes].unsqueeze(1)  # Shape: (N, 1, D)
        positives = self.embeddings[positive_samples]  # Shape: (N, P, D)
        negatives = self.embeddings[negative_samples]  # Shape: (N, Q, D)

        # Euclidean distance between anchors and positives and take mean
        pos_dist = torch.mean(torch.norm(anchors - positives, dim=-1), dim=-1)
        # Euclidean distance between anchors and negatives and take mean
        neg_dist = torch.mean(torch.norm(anchors - negatives, dim=-1), dim=-1)

        return pos_dist, neg_dist

    @time_it
    def store_subset(self):
        # store the subset of the idx in a dictionary
        sample_idx = torch.where(self.data.sample_mask)[0]
        subset_dict = {}
        for idx in sample_idx:
            idx_ = idx.reshape(-1)
            subset, _, _, _ = k_hop_subgraph(
                idx_, self.args.k_hop, self.data.edge_index
            )
            subset_set = set(subset.tolist())
            subset_dict[idx.item()] = subset_set
        self.subset_dict = subset_dict

    def store_edge_index_for_poison(self, data, idx, hop=1):
        edge_index_for_poison_dict = {}
        for idx in range(len(data.train_mask)):
            if data.retain_mask[idx]:
                _, edge_index_for_poison, _, _ = k_hop_subgraph(idx, hop, data.edge_index)
                edge_index_for_poison_dict[idx] = edge_index_for_poison
        self.edge_index_for_poison_dict = edge_index_for_poison_dict

    @time_it
    def get_distances_batch(self, batch_size=64):
        st = time.time()
        self.embeddings = self.model(self.data.x, self.data.edge_index)
        #print(f"Time taken to get embeddings: {time.time() - st}")

        num_masks = len(self.data.train_mask)
        pos_dist = torch.zeros(num_masks)
        neg_dist = torch.zeros(num_masks)

        pos_dist = pos_dist.to(device)
        neg_dist = neg_dist.to(device)

        sample_indices = torch.where(self.data.sample_mask)[0]
        num_samples = len(sample_indices)

        #print(f"Number of samples: {num_samples}")

        attacked_set = set(self.attacked_idx.tolist())

        st = time.time()
        calc_time = 0

        for i in range(0, num_samples, batch_size):
            batch_indices = sample_indices[i : i + batch_size]
            batch_size = len(batch_indices)

            batch_positive_samples = [
                list(self.subset_dict[idx.item()] - attacked_set)
                for idx in batch_indices
            ]
            batch_negative_samples = [list(attacked_set) for _ in range(batch_size)]

            # Pad and create dense batches
            max_pos = max(len(s) for s in batch_positive_samples)
            max_neg = max(len(s) for s in batch_negative_samples)

            batch_pos = torch.stack(
                [
                    torch.tensor(s + [0] * (max_pos - len(s)))
                    for s in batch_positive_samples
                ]
            )
            batch_neg = torch.stack(
                [
                    torch.tensor(s + [0] * (max_neg - len(s)))
                    for s in batch_negative_samples
                ]
            )

            st_2 = time.time()
            batch_pos_dist, batch_neg_dist = self.calc_distances(
                batch_indices, batch_pos, batch_neg
            )
            calc_time += time.time() - st_2

            pos_dist[batch_indices] = batch_pos_dist.to(pos_dist.device)
            neg_dist[batch_indices] = batch_neg_dist.to(neg_dist.device)

        #print(f"Average time taken to calculate distances: {calc_time/num_samples}")
        #print(f"Average time taken to get distances: {(time.time() - st)/num_samples}")

        return pos_dist, neg_dist

    def get_distances_edge(self, batch_size=64):
        # attacked edge index contains all the edges that were maliciously added
        self.embeddings = self.model(self.data.x, self.data.edge_index)
        num_masks = len(self.data.train_mask)
        pos_dist = torch.zeros(num_masks).to(device)
        neg_dist = torch.zeros(num_masks).to(device)

        sample_indices = torch.where(self.data.sample_mask)[0]
        num_samples = len(sample_indices)

        #Batchwise positive and negative samples created
        for i in range(0, num_samples, batch_size):
            batch_indices = sample_indices[i : i + batch_size]
            batch_size = len(batch_indices)
            
            batch_positive_samples = [
                list(self.subset_dict[idx.item()] - self.negative_sample_dict[idx.item()])
                for idx in batch_indices
            ]
            batch_negative_samples = [list(self.negative_sample_dict[idx.item()]) for idx in batch_indices]
            # Pad and create dense batches
            max_pos = max(len(s) for s in batch_positive_samples)
            max_neg = max(len(s) for s in batch_negative_samples)

            batch_pos = torch.stack(
                [
                    torch.tensor(s + [0] * (max_pos - len(s)))
                    for s in batch_positive_samples
                ]
            )
            batch_neg = torch.stack(
                [
                    torch.tensor(s + [0] * (max_neg - len(s)))
                    for s in batch_negative_samples
                ]
            )

            # st_2 = time.time()
            batch_pos_dist, batch_neg_dist = self.calc_distances(
                batch_indices, batch_pos, batch_neg
            )
            # calc_time += time.time() - st_2

            pos_dist[batch_indices] = batch_pos_dist.to(pos_dist.device)
            neg_dist[batch_indices] = batch_neg_dist.to(neg_dist.device)

        pos_dist = torch.tensor(pos_dist)
        neg_dist = torch.tensor(neg_dist)
        return pos_dist, neg_dist

    @time_it
    def get_model_embeddings(self):
        self.embeddings = self.model(self.data.x, self.data.edge_index)

    def train_node(self):
        self.model = self.model.to(device)
        self.data = self.data.to(device)
        args = self.args
        optimizer = self.optimizer
        # attacked idx must be a list of nodes
        for epoch in trange(
            args.contrastive_epochs_1 + args.contrastive_epochs_2, desc="Unlearning"
        ):
            self.model.train()
            self.embeddings = self.model(self.data.x, self.data.edge_index[:, self.data.dr_mask])
            if epoch <= args.contrastive_epochs_1:
                pos_dist, neg_dist = self.get_distances_batch()
                lmda = args.contrastive_lambda
                loss = self.unlearn_loss(
                    pos_dist, neg_dist, margin=args.contrastive_margin, lmda=lmda
                )
            else:
                pos_dist = None
                neg_dist = None
                lmda = 1
                loss = self.unlearn_loss(
                    pos_dist, neg_dist, margin=args.contrastive_margin, lmda=lmda
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def train_edge(self):
        # attack idx must be a list of tuples (u,v)
        args = self.args
        optimizer = self.optimizer
        
        for epoch in trange(
            args.contrastive_epochs_1 + args.contrastive_epochs_2, desc="Unlearning"
        ):
            self.model.train()
            self.embeddings = self.model(self.data.x, self.data.edge_index)
            if epoch <= args.contrastive_epochs_1:
                pos_dist, neg_dist = self.get_distances_edge()
                lmda = args.contrastive_lambda
            else:
                pos_dist = None
                neg_dist = None
                lmda = 1
            loss = self.unlearn_loss(
                pos_dist, neg_dist, margin=args.contrastive_margin, lmda=lmda
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.save_best()
        return

    def train(self):

        # attack_idx is an extra needed parameter which is defined above in both node and edge functions
        self.data.retain_mask = self.data.train_mask.clone()
        self.get_sample_points()
        self.store_subset()

        start_time = time.time()
        if self.args.request == "node":
            self.train_node()
        elif self.args.request == "edge":
            self.train_edge()
        end_time = time.time()
        train_acc, msc_rate, f1 = self.evaluate(is_dr=True, use_val=True)
        print(f"Train Acc: {train_acc}, Misclassification: {msc_rate},  F1 Score: {f1}")

        print(f"Training time: {end_time - start_time}")

        return train_acc, msc_rate, end_time - start_time