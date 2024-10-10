
import torch, torchmetrics, tqdm, copy, time
from torch.cuda.amp import autocast
import numpy as np
from torch.cuda.amp import GradScaler
from os import makedirs
from os.path import exists
from torch.nn import functional as F
import itertools

import random, torch, copy, tqdm
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from functools import partial
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from typing import Dict, List
from .base import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reference: https://github.com/if-loops/selective-synaptic-dampening/blob/main/src/forget_random_strategies.py
# Hessian based method that is more efficient than Fisher etc. and outperforms.

class ParameterPerturber:
    def __init__(
        self,
        model,
        opt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        parameters=None,
    ):
        self.model = model

        self.opt = opt
        self.device = device
        self.alpha = None
        self.xmin = None

        print(parameters)
        self.lower_bound = parameters["lower_bound"]
        self.exponent = parameters["exponent"]
        self.magnitude_diff = parameters["magnitude_diff"]  # unused
        self.min_layer = parameters["min_layer"]
        self.max_layer = parameters["max_layer"]
        self.forget_threshold = parameters["forget_threshold"] #unused
        self.dampening_constant = parameters["dampening_constant"] #lambda 
        self.selection_weighting = parameters["selection_weighting"] #alpha

    def zerolike_params_dict(self, model: torch.nn) -> Dict[str, torch.Tensor]:
        """
        Taken from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Returns a dict like named_parameters(), with zeroed-out parameter valuse
        Parameters:
        model (torch.nn): model to get param dict from
        Returns:
        dict(str,torch.Tensor): dict of zero-like params
        """
        return dict(
            [
                (k, torch.zeros_like(p, device=p.device))
                for k, p in model.named_parameters()
            ]
        )

    def calc_importance(self, data, mask) -> Dict[str, torch.Tensor]:
        """
        Calculate per-parameter importance for a GNN model using masked data.
        
        Parameters:
        data: The data object containing node features, labels, etc.
        mask: A boolean mask to select specific nodes.
        
        Returns:
        importances (dict(str, torch.Tensor([]))): Dictionary containing the importance for each parameter.
        """
        criterion = nn.CrossEntropyLoss()
        importances = self.zerolike_params_dict(self.model)
        
        # Select the masked data
        x_masked, y_masked = data.x[mask], data.y[mask]
        
        # Move the data to the appropriate device
        x_masked, y_masked = x_masked.to(self.device), y_masked.to(self.device)
        
        # Forward pass and compute gradients
        self.opt.zero_grad()
        out = self.model(data.x, data.edge_index[data.dr_mask])
        loss = criterion(out[mask], y_masked)
        loss.backward()

        # Accumulate the importance for each parameter
        for (k1, p), (k2, imp) in zip(self.model.named_parameters(), importances.items()):
            if p.grad is not None:
                imp.data += p.grad.data.clone().pow(2)
        
        # Average the importance over the number of masked nodes
        for _, imp in importances.items():
            imp.data /= float(mask.sum().item())
        
        return importances

    def modify_weight(
        self,
        original_importance: List[Dict[str, torch.Tensor]],
        forget_importance: List[Dict[str, torch.Tensor]],
    ) -> None:
        """
        Perturb weights based on the SSD equations given in the paper
        Parameters:
        original_importance (List[Dict[str, torch.Tensor]]): list of importances for original dataset
        forget_importance (List[Dict[str, torch.Tensor]]): list of importances for forget sample
        threshold (float): value to multiply original imp by to determine memorization.

        Returns:
        None

        """

        with torch.no_grad():
            for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                self.model.named_parameters(),
                original_importance.items(),
                forget_importance.items(),
            ):
                # Synapse Selection with parameter alpha
                oimp_norm = oimp.mul(self.selection_weighting)
                locations = torch.where(fimp > oimp_norm)

                # Synapse Dampening with parameter lambda
                weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(
                    self.exponent
                )
                update = weight[locations]
                # Bound by 1 to prevent parameter values to increase.
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound
                p[locations] = p[locations].mul(update)

class LinearLR(_LRScheduler):
    def __init__(self, optimizer, T, warmup_epochs=100, last_epoch=-1):
        self.T = float(T)
        self.warm_ep = warmup_epochs
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch - self.warm_ep >= 0:
            rate = (1 - ((self.last_epoch-self.warm_ep)/self.T))
        else:
            rate = (self.last_epoch+1)/(self.warm_ep+1)
        return [rate*base_lr for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        return self.get_lr()
    
class Naive(Trainer):
    def __init__(self, opt, model, data, optimizer):
        super().__init__(model, data, optimizer)
        self.opt = opt
        self.curr_step, self.best_top1 = 0, 0
        self.best_model = None
        self.data = data
        self.opt.training_epochs = opt.unlearning_epochs
        self.set_model(model)
        self.optimizer = self.optimizer
        self.scheduler = LinearLR(self.optimizer, T=self.opt.training_epochs*1.25, warmup_epochs=self.opt.training_epochs//100) # Spend 1% time in warmup, and stop 66% of the way through training 
        self.top1 = torchmetrics.Accuracy(task="multiclass", num_classes=self.data.num_classes).to(device)
        self.scaler = GradScaler()

    def set_model(self, model):
        self.model = model
        self.model.to(device)

    def forward_pass(self, images, target):
        output = self.model(images)
        loss = F.cross_entropy(output, target)
        self.top1(output, target)
        return loss

    def eval(self, data, mask, save_model=True, save_preds=False):
        
            self.model.eval()
            self.top1.reset()

            with torch.no_grad():
                # Select the masked data
                x_masked, y_masked = data.x[mask], data.y[mask]
                edge_index = data.edge_index
                
                # Move the data to the appropriate device
                x_masked, y_masked = x_masked.to(self.device), y_masked.to(self.device)

                # Forward pass through the model
                with autocast():
                    output = self.model(x_masked, edge_index[data.dr_mask])
                
                # Compute the evaluation metric (e.g., accuracy)
                self.top1(output, y_masked)

            top1 = self.top1.compute().item()
            self.top1.reset()
            print(f'Step: {self.curr_step} Val Top1: {top1*100:.2f}')
        
class SSDTrainer(Naive):
    def __init__(self, model, data, optimizer, opt ):
        super().__init__(opt, model, data, optimizer)

    def ssd_tuning(
        self,
        model,
        data,
        dampening_constant,
        selection_weighting,
        device,
        optimizer
    ):
        
        parameters = {
            "lower_bound": 1,
            "exponent": 1,
            "magnitude_diff": None,
            "min_layer": -1,
            "max_layer": -1,
            "forget_threshold": 1,
            "dampening_constant": dampening_constant,
            "selection_weighting": selection_weighting,
        }

        # load the trained model
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.015)
        pdr = ParameterPerturber(model, optimizer, device, parameters)
        model = model.eval()

        sample_importances = pdr.calc_importance(data, data.node_df_mask)

        original_importances = pdr.calc_importance(data, data.train_mask)
        pdr.modify_weight(original_importances, sample_importances)
        # self.eval(data, data.df_mask)
        # self.eval(data, data.train_mask)

        return model


    def unlearn(self):
        time_start = time.process_time()
        self.best_model = self.ssd_tuning(self.model, self.data, self.opt.SSDdampening, self.opt.SSDselectwt,device, self.optimizer)
        time_taken = time.process_time() - time_start
        return time_taken
    def train(self):
        time_taken = self.unlearn()
        train_acc, msc_rate, f1 = self.evaluate()
        print(f'Test Acc: {train_acc}, Misclassification: {msc_rate},  F1 Score: {f1}')
        return None, None, time_taken
        