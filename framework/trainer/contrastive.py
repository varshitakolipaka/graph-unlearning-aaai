import os, math
import copy
import time
#import wandb
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling, k_hop_subgraph
from torch_geometric.loader import GraphSAINTRandomWalkSampler

from .base import Trainer, KGTrainer, NodeClassificationTrainer
from ..evaluation import *
from ..utils import *

class ContrastiveNodeUnlearnTrainer(NodeClassificationTrainer):

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
        pos_dist = pos_dist[mask]
        neg_dist = neg_dist[mask]

        print(margin - neg_dist)

        pos_loss = torch.mean(pos_dist**2)
        neg_loss = torch.mean(F.relu(margin - neg_dist) ** 2)
        print(pos_loss, neg_loss)
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


    def calc_distance(self, model, data, node, positive_samples, negative_samples):
        # Contrastive loss
        embeddings = model.conv1(data.x, data.edge_index)
        embeddings = F.relu(embeddings)
        embeddings = F.dropout(embeddings, training=model.training)
        embeddings = model.conv2(embeddings, data.edge_index)

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


    def get_distances(self, model, data, attacked_set):
        # get pos, neg distances for nodes beforehand
        pos_dist = []
        neg_dist = []
        for idx in range(len(data.train_mask)):
            if data.retain_mask[idx]:
                # Get K-hop subgraph
                subset, edge_index, _, _ = k_hop_subgraph(idx, 2, data.edge_index)
                subset_set = set(subset.tolist())

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
                pos, neg = self.calc_distance(model, idx, positive_samples, negative_samples)
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
        for idx in range(len(data.train_mask)):
            # Get K-hop subgraph
            subset, edge_index, _, _ = k_hop_subgraph(idx, 2, data.edge_index)
            _, edge_index_for_poison, _, _ = k_hop_subgraph(
                idx, 1, data.edge_index
            )
            subset_set = set(subset.tolist())

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
            pos, neg = self.calc_distance(model, idx, positive_samples, negative_samples)
            pos_dist.append(pos.item())
            neg_dist.append(neg.item())

        # convert to tensor
        pos_dist = torch.tensor(pos_dist)
        neg_dist = torch.tensor(neg_dist)
        return pos_dist, neg_dist

    def train_node(self, model, data, optimizer, args, attacked_idx ,logits_ori=None, attack_model_all=None, attack_model_sub=None):
        
        # attacked idx must be a list of nodes
        
        model = model.to(device)
        data = data.to(device)
        
        # set a mask for the attacked nodes
        data.retain_mask = data.train_mask.clone()
        data.retain_mask[attacked_idx] = False
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

            print(f"Epoch {epoch+1}, Loss: {loss:.4f}", end="\r")
            
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

            print(f"Epoch {epoch+1}, Loss: {loss:.4f}", end="\r")
        
        # save
        ckpt = {
            'model_state': {k: v.to('cpu') for k, v in model.state_dict().items()},
            # 'optimizer_state': [optimizer[0].state_dict(), optimizer[1].state_dict()],
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))
        
    
    def train_edge(self, model, data, optimizer, args, attacked_idx ,logits_ori=None, attack_model_all=None, attack_model_sub=None):
        # attack idx must be a list of tuples (u,v)
        
        for epoch in range(args.contrastive_epochs_1):
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

            print(f"Epoch {epoch+1}, Loss: {loss:.4f}", end="\r")
    
    def train(self, model, data, optimizer, args, attacked_idx, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        
        # attack_idx is an extra needed parameter which is defined above in both node and edge functions
        
        if args.request == "node":
            self.train_node(model, data, optimizer, args, attacked_idx, logits_ori, attack_model_all, attack_model_sub)


