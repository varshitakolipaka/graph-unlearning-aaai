import os
import time
import wandb
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import GraphSAINTRandomWalkSampler

from .base import NodeClassificationTrainer, Trainer, KGTrainer
from ..evaluation import *
from ..utils import *


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class UtUTrainer(Trainer):

    # def freeze_unused_mask(self, model, edge_to_delete, subgraph, h):
    #     gradient_mask = torch.zeros_like(delete_model.operator)
    #     
    #     edges = subgraph[h]
    #     for s, t in edges:
    #         if s < t:
    #             gradient_mask[s, t] = 1
    #     gradient_mask = gradient_mask.to(device)
    #     model.operator.register_hook(lambda grad: grad.mul_(gradient_mask))
    
    def train(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        if 'ogbl' in self.args.dataset:
            args.eval_on_cpu = False
            return self.train_fullbatch(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

        else:
            return self.train_fullbatch(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

    def train_fullbatch(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        model = model.to(device)
        data = data.to(device)

        best_metric = 0
        loss_fct = nn.MSELoss()

        z = model(data.x, data.train_pos_edge_index[:, data.dr_mask])
        # Save
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
        torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))
        self.trainer_log['best_metric'] = best_metric
        print("TRAINING DONE")

    def train_fullbatch_node(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        model = model.to(device)
        data = data.to(device)

        best_metric = 0
        loss_fct = nn.MSELoss()

        z = model(data.x[data.dr_mask], data.edge_index)
        # Save
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
        torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))
        self.trainer_log['best_metric'] = best_metric


    def train_minibatch(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        start_time = time.time()
        best_metric = 0

        # Save models and node embeddings
        print('Saving final checkpoint')
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
        # torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))
        self.trainer_log['best_metric'] = best_metric
class UtUTrainerNode(NodeClassificationTrainer):

    # def freeze_unused_mask(self, model, edge_to_delete, subgraph, h):
    #     gradient_mask = torch.zeros_like(delete_model.operator)
    #     
    #     edges = subgraph[h]
    #     for s, t in edges:
    #         if s < t:
    #             gradient_mask[s, t] = 1
    #     gradient_mask = gradient_mask.to(device)
    #     model.operator.register_hook(lambda grad: grad.mul_(gradient_mask))
    
    # dr, df masks
    
    def train(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        if 'ogbl' in self.args.dataset:
            args.eval_on_cpu = False
            return self.train_fullbatch(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

        else:
            return self.train_fullbatch(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

    def train_fullbatch(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        model = model.to(device)
        data = data.to(device)

        best_metric = 0
        loss_fct = nn.MSELoss()
        
        print('shape: ', data.edge_index[:, data.dr_mask].shape)
        z = model(data.x, data.edge_index[:, data.dr_mask]) # df mask = edges connected to the node
        # Save 
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        print('storing at: ', args.checkpoint_dir)
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
        torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))
        self.trainer_log['best_metric'] = best_metric
        print("TRAINING DONE")

    # edge mask != node mask
    # train_mask
    # dr/df

    def train_fullbatch_node(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        model = model.to(device)
        data = data.to(device)

        best_metric = 0
        loss_fct = nn.MSELoss()

        z = model(data.x[data.dr_mask], data.edge_index)
        # Save
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
        torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))
        self.trainer_log['best_metric'] = best_metric


    def train_minibatch(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        start_time = time.time()
        best_metric = 0

        # Save models and node embeddings
        print('Saving final checkpoint')
        ckpt = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
        # torch.save(z, os.path.join(args.checkpoint_dir, 'node_embeddings.pt'))
        self.trainer_log['best_metric'] = best_metric


class KGRetrainTrainer(KGTrainer):
    pass