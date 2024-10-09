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
from torch_geometric.utils import (
    negative_sampling,
    k_hop_subgraph,
    to_scipy_sparse_matrix,
)
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch.optim.lr_scheduler import _LRScheduler
from .base import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_dataset_dict = {
    "Cora": {
        "class1": 5,
        "class2": 63,
    },
    "PubMed": {
        "class1": 2,
        "class2": 1,
    },
    "Amazon": {
        "class1": 3,
        "class2": 4,
    },
    "CS": {
        "class1": 3,
        "class2": 12,
    },
}

def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds to execute.")
        return result

    return wrapper


class LinearLR(_LRScheduler):
    r"""Set the learning rate of each parameter group with a linear
    schedule: :math:`\eta_{t} = \eta_0*(1 - t/T)`, where :math:`\eta_0` is the
    initial lr, :math:`t` is the current epoch or iteration (zero-based) and
    :math:`T` is the total training epochs or iterations. It is recommended to
    use the iteration based calculation if the total number of epochs is small.
    When last_epoch=-1, sets initial lr as lr.
    It is studied in
    `Budgeted Training: Rethinking Deep Neural Network Training Under Resource
     Constraints`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T (int): Total number of training epochs or iterations.
        last_epoch (int): The index of last epoch or iteration. Default: -1.

    .. _Budgeted Training\: Rethinking Deep Neural Network Training Under
    Resource Constraints:
        https://arxiv.org/abs/1905.04753
    """

    def __init__(self, optimizer, T, warmup_epochs=100, last_epoch=-1):
        self.T = float(T)
        self.warm_ep = warmup_epochs
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch - self.warm_ep >= 0:
            rate = 1 - ((self.last_epoch - self.warm_ep) / self.T)
        else:
            rate = (self.last_epoch + 1) / (self.warm_ep + 1)
        return [rate * base_lr for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        return self.get_lr()


def distill_kl_loss(y_s, y_t, T, reduction="sum"):
    p_s = torch.nn.functional.log_softmax(y_s / T, dim=1)
    p_t = torch.nn.functional.softmax(y_t / T, dim=1)
    loss = torch.nn.functional.kl_div(p_s, p_t, reduction=reduction)
    if reduction == "none":
        loss = torch.sum(loss, dim=1)
    loss = loss * (T**2) / y_s.shape[0]
    return loss


class ContrastiveAscentNoLinkTrainer(Trainer):
    def __init__(self, model, data, optimizer, args):
        super().__init__(model, data, optimizer, args)
        self.attacked_idx = data.attacked_idx
        self.embeddings = None
        self.criterion = torch.nn.CrossEntropyLoss()
        self.data.poison_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        self.data.poison_mask[self.data.poisoned_nodes] = True
        self.og_model = copy.deepcopy(model)
        self.og_model.eval()
        self.og_model.to(device)

    def reverse_features(self, features):
        reverse_features = features.clone()
        if self.args.request == "edge":
            for idx in self.data.poisoned_nodes:
                reverse_features[idx] = 1 - reverse_features[idx]
            return reverse_features

        for idx in self.attacked_idx:
            reverse_features[idx] = 1 - reverse_features[idx]
        return reverse_features

    def get_sample_points(self):
        if self.args.request == "edge":
            og_logits = F.softmax(self.model(self.data.x, self.data.edge_index), dim=1)
            temp_features = self.data.x.clone()
            reverse_feature = self.reverse_features(temp_features)
            final_logits = F.softmax(
                self.model(reverse_feature, self.data.edge_index), dim=1
            )
            diff = torch.abs(og_logits - final_logits)
            diff = torch.mean(diff, dim=1)
            diff = diff[self.data.poisoned_nodes]
            frac = self.args.contrastive_frac
            _, indices = torch.topk(
                diff, int(frac * len(self.data.poisoned_nodes)), largest=True
            )
            influence_nodes_with_unlearning_nodes = self.data.poisoned_nodes[indices]
            print(f"Nodes influenced: {len(influence_nodes_with_unlearning_nodes)}")
            self.data.sample_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
            self.data.sample_mask[influence_nodes_with_unlearning_nodes] = True

            poisoned_edges = self.data.edge_index[:, self.data.df_mask]
            negative_sample_dict = {int: set()}

            for i in range(len(poisoned_edges[0])):
                toNode = poisoned_edges[0][i].item()
                fromNode = poisoned_edges[1][i].item()

                if toNode not in negative_sample_dict:
                    negative_sample_dict[toNode] = set()
                negative_sample_dict[toNode].add(fromNode)

                if fromNode not in negative_sample_dict:
                    negative_sample_dict[fromNode] = set()
                negative_sample_dict[fromNode].add(toNode)
            self.negative_sample_dict = negative_sample_dict
            return

        subset, _, _, _ = k_hop_subgraph(
            self.attacked_idx.clone().detach(), self.args.k_hop, self.data.edge_index
        )

        # remove attacked nodes from the subset
        subset = subset[~np.isin(subset.cpu(), self.attacked_idx.cpu())]

        og_logits = F.softmax(self.model(self.data.x, self.data.edge_index), dim=1)
        temp_features = self.data.x.clone()
        reverse_feature = self.reverse_features(temp_features)
        final_logits = F.softmax(
            self.model(reverse_feature, self.data.edge_index), dim=1
        )

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

    def ascent_loss(self, mask):
        return -F.cross_entropy(self.embeddings[mask], self.data.y[mask])

    @time_it
    def task_loss(self, mask=None, ascent=False):
        # use the retain mask to calculate the loss
        if self.args.request == "edge":
            self.embeddings = self.model(
                self.data.x, self.data.edge_index[:, self.data.dr_mask]
            )

        if mask is None:
            try:
                mask = self.data.retain_mask
            except:
                mask = self.data.train_mask

        if ascent:
            mask = self.data.poison_mask
            return self.ascent_loss(mask)

        loss = self.criterion(self.embeddings[mask], self.data.y[mask])

        return loss

    @time_it
    def contrastive_loss(self, pos_dist, neg_dist, margin):
        pos_loss = torch.mean(pos_dist)
        neg_loss = F.relu(torch.mean((margin - neg_dist)))
        loss = pos_loss + neg_loss
        return loss
    
    def sage_loss(self, anchors, pos_embs, neg_embs):
        pos_loss = F.logsigmoid((anchors*pos_embs).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(anchors*neg_embs).sum(-1)).mean()
        
        return -pos_loss - neg_loss
        

    def kd_loss(self):
        with torch.no_grad():
            self.og_embeddings = self.og_model(self.data.x, self.data.edge_index)
        kd_loss = self.args.scrubAlpha * distill_kl_loss(
            self.embeddings[self.data.retain_mask],
            self.og_embeddings[self.data.retain_mask],
            self.args.kd_T,
        )
        return kd_loss

    @time_it
    def unlearn_loss(self, pos_dist, neg_dist, margin=1.0, lmda=0.8, ascent=False):
        if lmda == 1:
            return self.task_loss()

        if lmda == 0:
            return self.contrastive_loss(pos_dist, neg_dist, margin)

        return lmda * self.task_loss(ascent=ascent) + (
            1 - lmda
        ) * self.contrastive_loss(pos_dist, neg_dist, margin)

    def calc_distances(self, nodes, positive_samples, negative_samples):
        # Vectorized contrastive loss calculation

        anchors = self.embeddings[nodes].unsqueeze(1)  # Shape: (N, 1, D)
        positives = self.embeddings[positive_samples] * self.mask_pos  # Shape: (N, P, D)
        negatives = self.embeddings[negative_samples] * self.mask_neg # Shape: (N, Q, D)

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
                _, edge_index_for_poison, _, _ = k_hop_subgraph(
                    idx, hop, data.edge_index
                )
                edge_index_for_poison_dict[idx] = edge_index_for_poison
        self.edge_index_for_poison_dict = edge_index_for_poison_dict

    @time_it
    def get_distances_batch(self, batch_size=64):
        st = time.time()
        # print(f"Time taken to get embeddings: {time.time() - st}")

        num_masks = len(self.data.train_mask)
        pos_dist = torch.zeros(num_masks)
        neg_dist = torch.zeros(num_masks)

        pos_dist = pos_dist.to(device)
        neg_dist = neg_dist.to(device)

        sample_indices = torch.where(self.data.sample_mask)[0]
        num_samples = len(sample_indices)

        # print(f"Number of samples: {num_samples}")

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
                
            self.mask_pos = torch.stack(
                [
                    torch.tensor([1] * len(s) + [0] * (max_pos - len(s)))
                    for s in batch_positive_samples
                ]
            ).float().unsqueeze(-1).to(device)

            self.mask_neg = torch.stack(
                [
                    torch.tensor([1] * len(s) + [0] * (max_neg - len(s)))
                    for s in batch_negative_samples
                ]
            ).float().unsqueeze(-1).to(device)

            st_2 = time.time()
            try:
                batch_pos_dist, batch_neg_dist = self.calc_distances(
                    batch_indices, batch_pos, batch_neg
                )
                calc_time += time.time() - st_2
            except:
                continue

            pos_dist[batch_indices] = batch_pos_dist.to(pos_dist.device)
            neg_dist[batch_indices] = batch_neg_dist.to(neg_dist.device)

        # print(f"Average time taken to calculate distances: {calc_time/num_samples}")
        # print(f"Average time taken to get distances: {(time.time() - st)/num_samples}")

        return pos_dist, neg_dist
    
    def run_sage_batch(self, batch_size=128):
        st = time.time()

        sample_indices = torch.where(self.data.sample_mask)[0]
        num_samples = len(sample_indices)

        attacked_set = set(self.attacked_idx.tolist())
        attacked_list = list(attacked_set)

        total_loss = 0
        calc_time = 0

        for i in range(0, num_samples, batch_size):
            batch_indices = sample_indices[i : i + batch_size]
            batch_size = len(batch_indices)

            # Vectorize batch_positive_samples
            batch_positive_samples = [
                list(self.subset_dict[idx.item()] - attacked_set) 
                for idx in batch_indices
            ]
            
            # Max lengths can be computed once per loop iteration
            max_pos = max(len(s) for s in batch_positive_samples)
            max_neg = len(attacked_list)  # Always fixed since attacked_set size won't change

            # Preallocate memory for tensors
            batch_pos = torch.zeros((batch_size, max_pos), dtype=torch.long)
            batch_neg = torch.zeros((batch_size, max_neg), dtype=torch.long)
            mask_pos = torch.zeros((batch_size, max_pos), dtype=torch.float32)
            mask_neg = torch.ones((batch_size, max_neg), dtype=torch.float32)  # Fixed

            for idx, pos_samples in enumerate(batch_positive_samples):
                pos_len = len(pos_samples)
                batch_pos[idx, :pos_len] = torch.tensor(pos_samples)
                mask_pos[idx, :pos_len] = 1.0

            # Move everything to the correct device
            device = self.embeddings.device
            batch_pos, batch_neg = batch_pos.to(device), batch_neg.to(device)
            mask_pos, mask_neg = mask_pos.to(device).unsqueeze(-1), mask_neg.to(device).unsqueeze(-1)

            st_2 = time.time()
            try:
                anchor_embs = self.embeddings[batch_indices].unsqueeze(1)
                pos_embs = self.embeddings[batch_pos] * mask_pos
                neg_embs = self.embeddings[batch_neg] * mask_neg

                batch_loss = self.sage_loss(anchor_embs, pos_embs, neg_embs)
                calc_time += time.time() - st_2
            except Exception as e:
                continue

            total_loss += batch_loss

        return total_loss


    def get_distances_edge(self, batch_size=64):
        # attacked edge index contains all the edges that were maliciously added
        num_masks = len(self.data.train_mask)
        pos_dist = torch.zeros(num_masks).to(device)
        neg_dist = torch.zeros(num_masks).to(device)

        sample_indices = torch.where(self.data.sample_mask)[0]
        num_samples = len(sample_indices)

        # Batchwise positive and negative samples created
        for i in range(0, num_samples, batch_size):
            batch_indices = sample_indices[i : i + batch_size]
            batch_size = len(batch_indices)

            batch_positive_samples = [
                list(
                    self.subset_dict[idx.item()] - self.negative_sample_dict[idx.item()]
                )
                for idx in batch_indices
            ]
            batch_negative_samples = [
                list(self.negative_sample_dict[idx.item()]) for idx in batch_indices
            ]
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
            
            self.mask_pos = torch.stack(
                [
                    torch.tensor([1] * len(s) + [0] * (max_pos - len(s)))
                    for s in batch_positive_samples
                ]
            ).float().unsqueeze(-1).to(device)

            self.mask_neg = torch.stack(
                [
                    torch.tensor([1] * len(s) + [0] * (max_neg - len(s)))
                    for s in batch_negative_samples
                ]
            ).float().unsqueeze(-1).to(device)

            # st_2 = time.time()
            batch_pos_dist, batch_neg_dist = self.calc_distances(
                batch_indices, batch_pos, batch_neg
            )
            # calc_time += time.time() - st_2

            pos_dist[batch_indices] = batch_pos_dist.to(pos_dist.device)
            neg_dist[batch_indices] = batch_neg_dist.to(neg_dist.device)

        return pos_dist, neg_dist
    
    def run_sage_batch_edge(self, batch_size=64):
        # attacked edge index contains all the edges that were maliciously added
        num_masks = len(self.data.train_mask)

        sample_indices = torch.where(self.data.sample_mask)[0]
        num_samples = len(sample_indices)

        # Batchwise positive and negative samples created
        total_loss = 0
        for i in range(0, num_samples, batch_size):
            batch_indices = sample_indices[i : i + batch_size]
            batch_size = len(batch_indices)

            batch_positive_samples = [
                list(
                    self.subset_dict[idx.item()] - self.negative_sample_dict[idx.item()]
                )
                for idx in batch_indices
            ]
            batch_negative_samples = [
                list(self.negative_sample_dict[idx.item()]) for idx in batch_indices
            ]
            
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
            
            self.mask_pos = torch.stack(
                [
                    torch.tensor([1] * len(s) + [0] * (max_pos - len(s)))
                    for s in batch_positive_samples
                ]
            ).float().unsqueeze(-1).to(device)

            self.mask_neg = torch.stack(
                [
                    torch.tensor([1] * len(s) + [0] * (max_neg - len(s)))
                    for s in batch_negative_samples
                ]
            ).float().unsqueeze(-1).to(device)

            try:
                anchor_embs = self.embeddings[batch_indices].unsqueeze(1)
                pos_embs = self.embeddings[batch_pos] * self.mask_pos
                neg_embs = self.embeddings[batch_neg] * self.mask_neg
                batch_loss = self.sage_loss(anchor_embs, pos_embs, neg_embs)
            except:
                continue
            
            total_loss += batch_loss

        return total_loss

    @time_it
    def get_model_embeddings(self):
        self.embeddings = self.model(self.data.x, self.data.edge_index)

    def train_node(self):
        self.model = self.model.to(device)
        self.data = self.data.to(device)
        args = self.args
        optimizer = self.optimizer

        ascent_optimizer = torch.optim.Adam(self.model.parameters(), lr=args.ascent_lr)

        descent_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.descent_lr
        )

        # attacked idx must be a list of nodes
        for epoch in trange(args.steps, desc="Unlearning"):
            self.save_best()
            for i in range(args.contrastive_epochs_1 + args.contrastive_epochs_2):
                iter_start_time = time.time()
                self.model.train()

                self.embeddings = self.model(
                    self.data.x, self.data.edge_index
                )
                if i < args.contrastive_epochs_1:
                    optimizer.zero_grad()
                    # pos_dist, neg_dist = self.get_distances_batch()
                    # loss = self.contrastive_loss(
                    #     pos_dist, neg_dist, margin=args.contrastive_margin
                    # )
                    loss = self.run_sage_batch()

                    loss.backward()
                    optimizer.step()
                else:
                    ascent_optimizer.zero_grad()

                    ascent_loss = self.ascent_loss(self.data.poison_mask)

                    ascent_loss.backward()
                    ascent_optimizer.step()
                    # ascent_scheduler.step()

                    descent_optimizer.zero_grad()

                    self.embeddings = self.model(
                        self.data.x, self.data.edge_index[:, self.data.dr_mask]
                    )
                    # self.embeddings = self.model(
                    #     self.data.x, self.data.edge_index # TESTING
                    # )
                    
                    finetune_loss = F.cross_entropy(
                        self.embeddings[self.data.retain_mask],
                        self.data.y[self.data.retain_mask],
                    )
                    # kd_loss = self.kd_loss()
                    descent_loss = finetune_loss

                    descent_loss.backward()
                    descent_optimizer.step()
                    # descent_scheduler.step()
                    
                # save best model
                self.unlearning_time += time.time() - iter_start_time
                cutoff = self.save_best(is_dr=True)
                if cutoff:
                    self.load_best()
                    return
        
        # load best model
        self.load_best()

    def train_edge(self):
        # attack idx must be a list of tuples (u,v)
        args = self.args
        optimizer = self.optimizer
        
        ascent_optimizer = torch.optim.Adam(self.model.parameters(), lr=args.ascent_lr)

        descent_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.descent_lr
        )

        for epoch in trange(args.steps, desc="Unlearning"):
            for i in range(args.contrastive_epochs_1 + args.contrastive_epochs_2):
                iter_start_time = time.time()
                self.model.train()
                optimizer.zero_grad()

                self.embeddings = self.model(
                    self.data.x, self.data.edge_index
                )
                if i < args.contrastive_epochs_1:
                    # pos_dist, neg_dist = self.get_distances_edge()
                    # lmda = 0
                    # loss = self.unlearn_loss(
                    #     pos_dist, neg_dist, margin=args.contrastive_margin, lmda=lmda
                    # )
                    loss = self.run_sage_batch_edge()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    ascent_optimizer.zero_grad()

                    ascent_loss = self.ascent_loss(self.data.poison_mask)

                    ascent_loss.backward()
                    ascent_optimizer.step()
                    # ascent_scheduler.step()

                    descent_optimizer.zero_grad()

                    if self.args.linked:
                        self.embeddings = self.model(
                            self.data.x, self.data.edge_index
                        )
                    else:
                        self.embeddings = self.model(
                            self.data.x, self.data.edge_index[:, self.data.dr_mask]
                        )

                    finetune_loss = F.cross_entropy(
                        self.embeddings[self.data.retain_mask],
                        self.data.y[self.data.retain_mask],
                    )
                    # kd_loss = self.kd_loss()
                    # descent_loss = finetune_loss + kd_loss

                    finetune_loss.backward()
                    descent_optimizer.step()
                    # descent_scheduler.step()

                self.unlearning_time += time.time() - iter_start_time
                # save best model
                if i % 2 == 0:
                    if self.args.linked:
                        cutoff = self.save_best(is_dr=False)
                    else:
                        cutoff = self.save_best()
                        if cutoff:
                            break
            
        # load best model
        self.load_best()

    def train(self):

        # attack_idx is an extra needed parameter which is defined above in both node and edge functions
        self.data.retain_mask = self.data.train_mask.clone()
        self.get_sample_points()
        self.store_subset()

        self.start_time = time.time()
        self.best_model_time = self.start_time
        if self.args.request == "node":
            self.train_node()
        elif self.args.request == "edge":
            self.train_edge()
        
        if self.args.linked:
            is_dr = False
        else:
            is_dr = True
        train_acc, msc_rate, f1 = self.evaluate(is_dr=is_dr, use_val=True)

        print(f"Training time: {self.best_model_time}, Train Acc: {train_acc}, Msc Rate: {msc_rate}, F1: {f1}")
        forg, util, forg_f1, util_f1 = self.get_score(self.args.attack_type, class1=class_dataset_dict[self.args.dataset]["class1"], class2=class_dataset_dict[self.args.dataset]["class2"])
        print(f"Forgotten: {forg}, Util: {util}, Forg F1: {forg_f1}, Util F1: {util_f1}")        

        return train_acc, msc_rate, self.best_model_time
