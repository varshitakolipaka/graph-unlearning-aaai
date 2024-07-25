import numpy as np
import torch
import networkx as nx
#import wandb
import os
from torch.nn import functional as F


#def #wandb_log(log_dict):
    #wandb.log({key.replace("_", "/", 1): value for key, value in log_dict.items()})

def get_node_edge(graph):
    degree_sorted_ascend = sorted(graph.degree, key=lambda x: x[1])

    return degree_sorted_ascend[-1][0]

def h_hop_neighbor(G, node, h):
    path_lengths = nx.single_source_dijkstra_path_length(G, node)
    return [node for node, length in path_lengths.items() if length == h]

def get_enclosing_subgraph(graph, edge_to_delete):
    subgraph = {0: [edge_to_delete]}
    s, t = edge_to_delete

    neighbor_s = []
    neighbor_t = []
    for h in range(1, 2+1):
        neighbor_s += h_hop_neighbor(graph, s, h)
        neighbor_t += h_hop_neighbor(graph, t, h)

        nodes = neighbor_s + neighbor_t + [s, t]

        subgraph[h] = list(graph.subgraph(nodes).edges())

    return subgraph

@torch.no_grad()
def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=pos_edge_index.device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

@torch.no_grad()
def get_link_labels_kg(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=pos_edge_index.device)
    link_labels[:pos_edge_index.size(1)] = 1.

    return link_labels

@torch.no_grad()
def negative_sampling_kg(edge_index, edge_type):
    '''Generate negative samples but keep the node type the same'''

    edge_index_copy = edge_index.clone()
    for et in edge_type.unique():
        mask = (edge_type == et)
        old_source = edge_index_copy[0, mask]
        new_index = torch.randperm(old_source.shape[0])
        new_source = old_source[new_index]
        edge_index_copy[0, mask] = new_source

    return edge_index_copy

def get_run_name(args):
    model_unlearn_pair = args.gnn + '/' + args.unlearning_model
    name = '_'.join([args.dataset, model_unlearn_pair, args.df+str(args.df_size)])
    if args.suffix != None:
        name += '_{}'.format(args.suffix)
    return name


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    From https://github.com/Bjarten/early-stopping-pytorch.
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, score, model, z=None):

        # score = -score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, z)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, z)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, z):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        ckpt = {
            'model_state': model.state_dict(),
            # 'optimizer_state': [optimizer[0].state_dict(), optimizer[1].state_dict()],
        }
        torch.save(z, os.path.join(self.path, 'node_embeddings.pt'))
        torch.save(ckpt, os.path.join(self.path, 'model_best.pt'))
        self.val_loss_min = val_loss

import random, torch, copy, tqdm
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from functools import partial
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from typing import Dict, List


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

    def calc_importance(self, dataloader: DataLoader=None, data=None, mask=None) -> Dict[str, torch.Tensor]:
        """
        Adapated from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Calculate per-parameter, importance
            returns a dictionary [param_name: list(importance per parameter)]
        Parameters:
        DataLoader (DataLoader): DataLoader to be iterated over
        Returns:
        importances (dict(str, torch.Tensor([]))): named_parameters-like dictionary containing list of importances for each parameter
        """
        criterion = nn.CrossEntropyLoss()
        importances = self.zerolike_params_dict(self.model)
        if dataloader:
            for data in tqdm.tqdm(dataloader):
                self.opt.zero_grad()

                out = self.model(data)
                loss = criterion(out, data.y)
                loss.backward()

                for (k1, p), (k2, imp) in zip(
                    self.model.named_parameters(), importances.items()
                ):
                    if p.grad is not None:
                        imp.data += p.grad.data.clone().pow(2)
        else:
            self.opt.zero_grad()
            out = self.model(data)
            loss = criterion(out[mask], data.y[mask])
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                self.model.named_parameters(), importances.items()
            ):
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        if dataloader:
            for _, imp in importances.items():
                imp.data /= float(len(dataloader))
        else:
            for _, imp in importances.items():
                imp.data /= float(len(data))
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


# default values:
# "dampening_constant" lambda: 1,
# "selection_weighting" alpha: 10 * model_size_scaler,
# model_size_scaler = 1
# if args.net == "ViT":
#     model_size_scaler = 0.5

#We found hyper-parameters using 50
# runs of the TPE search from Optuna (Akiba et al. 2019), for
# values α ∈ [0.1, 100]) and λ ∈ [0.1, 5]. We only conducted
# this search for the Rocket and Veh2 classes. We use λ=1
# and α=10 for all ResNet18 CIFAR tasks. For PinsFaceRecognition, we use α=50 and λ=0.1 due to the much greater
# similarity between classes. ViT also uses λ=1 on all CIFAR
# tasks. We change α=10 to α=5 for slightly improved performance on class and α=25 on sub-class unlearning.
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
            rate = (1 - ((self.last_epoch-self.warm_ep)/self.T))
        else:
            rate = (self.last_epoch+1)/(self.warm_ep+1)
        return [rate*base_lr for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        return self.get_lr()


def cutmix(x, y, alpha=1.0):
    assert(alpha > 0)
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam



def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def seed_everything(seed):
    """
    Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random

    Args:
        seed: the integer value seed for global random state
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed



class Model_with_TTA(torch.nn.Module):
    def __init__(self, model, mult_fact=1, tta_type='flip'):
        super(Model_with_TTA, self).__init__()
        self.model = model
        self.mult_fact = mult_fact
        self.tta_type = tta_type

    def forward(self, x):
        out = self.model(x)*self.mult_fact
        if self.tta_type == 'flip':
            out += self.model(torch.flip(x, dims=[3]))
            out /= 2
        return self.net(out)


def get_targeted_classes(dataset):
  if dataset == 'CIFAR10':
    classes = (3,5)
  elif dataset == 'CIFAR100':
    classes = (47,53)
  elif dataset in ['PCAM', 'DermNet', 'Pneumonia']:
    classes = (0,1)
  elif dataset in ['LFWPeople','CelebA']:
    # Raise NotImplemented Error
    assert(False), 'Not Implemented Yet'
  return classes


def unlearn_func(model, method, factor=0.1, device='cpu'):
  model = copy.deepcopy(model)
  model = model.cpu()
  if method == 'EU':
    model.apply(initialize_weights)
  elif method == 'Mixed':
    partialfunc = partial(modify_weights, factor=factor)
    model.apply(partialfunc)
  else:
    pass
  model.to(device)
  return model


def initialize_weights(m):
  if isinstance(m, torch.nn.Conv2d):
      m.reset_parameters()
      if m.bias is not None:
        torch.nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, torch.nn.BatchNorm2d):
      m.reset_parameters()
  elif isinstance(m, torch.nn.Linear):
      m.reset_parameters()
      if m.bias is not None:
        torch.nn.init.constant_(m.bias.data, 0)


def modify_weights(m, factor=0.1):
    if isinstance(m, torch.nn.Conv2d):
        m.weight.data = m.weight.data*factor
        if m.bias is not None:
            m.bias.data = m.bias.data*factor
    elif isinstance(m, torch.nn.BatchNorm2d):
        if m.affine:
            m.weight.data = m.weight.data*factor
            m.bias.data = m.bias.data*factor
    elif isinstance(m, torch.nn.Linear):
        m.weight.data = m.weight.data*factor
        if m.bias is not None:
            m.bias.data = m.bias.data*factor


# def distill_kl_loss(y_s, y_t, T, reduction='sum'):
#     p_s = torch.nn.functional.log_softmax(y_s/T, dim=1)
#     p_t = torch.nn.functional.softmax(y_t/T, dim=1)
#     loss = torch.nn.functional.kl_div(p_s, p_t, reduction=reduction)
#     if reduction == 'none':
#         loss = torch.sum(loss, dim=1)
#     loss = loss * (T**2) / y_s.shape[0]
#     return loss

def distill_kl_loss(y_s, y_t, T):
    p_s = torch.nn.functional.log_softmax(y_s / T, dim=0)  # changed dim to 0 for 1D tensor
    p_t = torch.nn.functional.softmax(y_t / T, dim=0)  # changed dim to 0 for 1D tensor
    loss = F.kl_div(p_s, p_t, reduction='batchmean') * (T ** 2)
    return loss


def compute_accuracy(preds, y):
    return np.equal(np.argmax(preds, axis=1), y).mean()

class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)