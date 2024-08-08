import os, math
import copy
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

    # def get_sample_points(self):
    #     # get the k-hop subgraph of attacked nodes, and add all the nodes in the subgraph to the sample_mask
    #     sample_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
    #     for idx in self.attacked_idx:
    #         idx_int = int(idx)
    #         subset, _, _, _ = k_hop_subgraph(
    #             idx_int, self.args.k_hop, self.data.edge_index
    #         )
    #         sample_mask[subset] = True

    #     #print(f"Number of nodes in the sampling: {sample_mask.sum().item()}")

    #     eps = self.args.contrastive_eps
    #     # add eps fraction of non-neighbor training nodes to the sample_mask
    #     num_to_add = int(eps * self.data.train_mask.sum().item())
    #     # TODO: check if the non-neighbors are being sampled correctly

    #     self.data.sample_mask = sample_mask

    def propagate(self, features, k, adj_norm):
        feature_list = []

        features= features.cpu()
        adj_norm= adj_norm.cpu()
        feature_list.append(features)

        for i in range(k):
            feature_list.append(torch.spmm(adj_norm, feature_list[-1]))
        return feature_list[-1]


    def reverse_features(self, features):
        reverse_features = features.clone()
        for idx in self.attacked_idx:
            reverse_features[idx] = 1-reverse_features[idx]

        return reverse_features

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float)

    def normalize_adj(self, adj, r=0.5):
        adj = adj + sp.eye(adj.shape[0])
        degrees = np.array(adj.sum(1))
        r_inv_sqrt_left = np.power(degrees, r - 1).flatten()
        r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
        r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)

        r_inv_sqrt_right = np.power(degrees, -r).flatten()
        r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
        r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)

        adj_normalized = adj.dot(r_mat_inv_sqrt_left).transpose().dot(r_mat_inv_sqrt_right)
        return adj_normalized

    #MEGU HIN sampling
    def get_sample_points(self):
        self.adj = self.sparse_mx_to_torch_sparse_tensor(self.normalize_adj(to_scipy_sparse_matrix(self.data.edge_index)))
        temp_features = self.data.x.clone()
        pfeatures = self.propagate(temp_features, self.args.k_hop, self.adj)
        reverse_feature = self.reverse_features(temp_features)
        re_pfeatures = self.propagate(reverse_feature, self.args.k_hop, self.adj)

        cos = nn.CosineSimilarity()
        sim = cos(pfeatures, re_pfeatures)

        alpha = 0.1
        gamma = 0.1
        max_val = 0.
        while True:
            influence_nodes_with_unlearning_nodes = torch.nonzero(sim <= alpha).flatten().cpu()
            if len(influence_nodes_with_unlearning_nodes.view(-1)) > 0:
                temp_max = torch.max(sim[influence_nodes_with_unlearning_nodes])
            else:
                alpha = alpha + gamma
                continue

            if temp_max == max_val:
                break

            max_val = temp_max
            alpha = alpha + gamma

        neighborkhop, _, _, two_hop_mask = k_hop_subgraph(
            torch.tensor(self.attacked_idx),
            self.args.k_hop,
            self.data.edge_index,
            num_nodes=self.data.num_nodes)

        neighborkhop = neighborkhop[~np.isin(neighborkhop.cpu(), self.attacked_idx)]
        neighbor_nodes = []
        for idx in influence_nodes_with_unlearning_nodes:
            if idx in neighborkhop and idx not in self.attacked_idx:
                neighbor_nodes.append(idx.item())

        self.data.sample_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), neighbor_nodes))


        # get the k-hop subgraph of attacked nodes, and add all the nodes in the subgraph to the sample_mask
        sample_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        for idx in self.attacked_idx:
            idx_int = int(idx)
            subset, _, _, _ = k_hop_subgraph(
                idx_int, self.args.k_hop, self.data.edge_index
            )
            sample_mask[subset] = True

        #print(f"Number of nodes in the sampling: {sample_mask.sum().item()}")

        eps = self.args.contrastive_eps
        # add eps fraction of non-neighbor training nodes to the sample_mask
        num_to_add = int(eps * self.data.train_mask.sum().item())
        # TODO: check if the non-neighbors are being sampled correctly

        self.data.sample_mask = sample_mask

    @time_it
    def task_loss(self):
        # use the retain mask to calculate the loss
        try:
            mask = self.data.retain_mask
        except:
            mask = self.data.train_mask
        loss = self.criterion(self.embeddings[mask], self.data.y[mask])
        return loss

    @time_it
    def contrastive_loss(self, pos_dist, neg_dist, margin):
        pos_loss = torch.mean(pos_dist)
        neg_loss = torch.mean(F.relu(margin - neg_dist))
        loss = pos_loss + neg_loss
        return loss

    @time_it
    def unlearn_loss(self, pos_dist, neg_dist, margin=1.0, lmda=0.8):
        if lmda == 1:
            return self.task_loss()
        return lmda * self.task_loss() + (1 - lmda) * self.contrastive_loss(
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
        subset_dict = {}
        for idx in trange(len(self.data.sample_mask), desc="Storing Subset"):
            if self.data.sample_mask[idx]:
                subset, _, _, _ = k_hop_subgraph(
                    idx, self.args.k_hop, self.data.edge_index
                )
                subset_set = set(subset.tolist())
                subset_dict[idx] = subset_set
        self.subset_dict = subset_dict

    def store_edge_index_for_poison(self, data, idx):
        edge_index_for_poison_dict = {}
        for idx in range(len(data.train_mask)):
            if data.retain_mask[idx]:
                _, edge_index_for_poison, _, _ = k_hop_subgraph(idx, 1, data.edge_index)
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
            if data.train_mask[idx]:
                # Get K-hop subgraph
                subset_set = self.subset_dict[idx]
                edge_index_for_poison = self.edge_index_for_poison_dict[idx]

                # convert edge_index to a list of tuples
                edge_index_list = [
                    (
                        edge_index_for_poison[0][i].item(),
                        edge_index_for_poison[1][i].item(),
                    )
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
                pos, neg = self.calc_distance(
                    embeddings, idx, positive_samples, negative_samples
                )
                pos_dist.append(pos.item())
                neg_dist.append(neg.item())
            else:
                pos_dist.append(0)
                neg_dist.append(0)

        # convert to tensor
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
            self.embeddings = self.model(self.data.x, self.data.edge_index)
            if epoch <= args.contrastive_epochs_1:
                pos_dist, neg_dist = self.get_distances_batch()
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

    def train_edge(self):
        # attack idx must be a list of tuples (u,v)
        model = self.model
        data = self.data
        args = self.args
        attacked_idx = self.attacked_idx
        optimizer = self.optimizer

        for epoch in trange(args.contrastive_epochs_1, desc="Unlearning 1"):
            pos_dist, neg_dist = self.get_distances_edge(model, data, attacked_idx)
            loss = self.unlearn_loss(
                model,
                data,
                pos_dist,
                neg_dist,
                margin=args.contrastive_margin,
                lmda=args.contrastive_lambda,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for epoch in trange(args.contrastive_epochs_2, desc="Unlearning 2"):
            loss = self.unlearn_loss(
                model, data, None, None, margin=args.contrastive_margin, lmda=1
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def train(self):

        # attack_idx is an extra needed parameter which is defined above in both node and edge functions
        self.data.retain_mask = self.data.train_mask.clone()
        self.get_sample_points()
        self.store_subset()
        start_time = time.time()
        if self.args.request == "node":
            self.train_node()
        elif self.args.request == "edge":
            self.store_edge_index_for_poison(self.data, self.attacked_idx)
            self.train_edge()
        end_time = time.time()
        train_acc, msc_rate, f1 = self.evaluate(is_dr=True)
        print(f"Train Acc: {train_acc}, Misclassification: {msc_rate},  F1 Score: {f1}")

        print(f"Training time: {end_time - start_time}")

        return train_acc, msc_rate, end_time - start_time
