import os
import time
#import wandb
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import GraphSAINTRandomWalkSampler

from .base import Trainer
from .base import EdgeTrainer, member_infer_attack

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
class UtUTrainer(Trainer):
    def __init__(self, model, data, optimizer, args):
        super().__init__(model, data, optimizer)
        self.args= args

    def train(self):
        # no training for UtU, only inference
        self.model = self.model.to(device)
        self.data = self.data.to(device)
        
        test_acc, msc_rate, f1 = self.evaluate(is_dr=True)
        print(f'Train Acc: {test_acc}, Misclassification: {msc_rate},  F1 Score: {f1}')

class UtUEdgeTrainer(EdgeTrainer):
    def __init__(self, model, data, optimizer,args):
        super().__init__(args)
        self.args= args

    def train(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        model = model.to(device)
        data = data.to(device)

        best_metric = 0
        loss_fct = nn.MSELoss()

        z = model(data.x, data.train_pos_edge_index[:, data.dr_mask])
        test_results = self.test(model, data, attack_model_all=attack_model_all, attack_model_sub=attack_model_sub)
        print('===AFTER UNLEARNING===', test_results[-1])
