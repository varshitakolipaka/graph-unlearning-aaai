import torch, torchmetrics, tqdm, copy, time
import numpy as np
from os import makedirs
from os.path import exists
from torch.nn import functional as F
from framework.training_args import parse_args
opt = parse_args()
from .base import Trainer
from torch.optim.lr_scheduler import _LRScheduler

def distill_kl_loss(y_s, y_t, T, reduction='sum'):
    p_s = torch.nn.functional.log_softmax(y_s/T, dim=1)
    p_t = torch.nn.functional.softmax(y_t/T, dim=1)
    loss = torch.nn.functional.kl_div(p_s, p_t, reduction=reduction)
    if reduction == 'none':
        loss = torch.sum(loss, dim=1)
    loss = loss * (T**2) / y_s.shape[0]
    return loss


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
    
class ScrubTrainer(Trainer):
    def __init__(self, model, poisoned_dataset, optimizer, opt):
        super().__init__(model, poisoned_dataset, optimizer)
        self.opt = opt
        self.opt.unlearn_iters = opt.unlearn_iters
        self.best_model = None
        self.curr_step = 0
        self.set_model(model)
        self.poisoned_dataset = poisoned_dataset


        self.scheduler = LinearLR(self.optimizer, T=self.opt.unlearn_iters*1.25, warmup_epochs=self.opt.unlearn_iters//100) # Spend 1% time in warmup, and stop 66% of the way through training 
        self.og_model = copy.deepcopy(model)
        self.og_model.eval()
        opt.unlearn_iters = opt.unlearn_iters
        self.opt.unlearn_iters = opt.unlearn_iters

    def set_model(self, model):
        self.model = model
        self.model

    def train_one_epoch(self, data, mask):
        self.model.train()

        if self.curr_step <= self.opt.unlearn_iters:
            self.optimizer.zero_grad()
            loss = self.forward_pass(data, mask)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.curr_step += 1

        return

    def forward_pass(self, data, mask):
        
        output = self.model(data.x, data.edge_index)

        with torch.no_grad():
            logit_t = self.og_model(data.x, data.edge_index)

        loss = F.cross_entropy(output[mask], data.y[mask])
        loss += self.opt.scrubAlpha * distill_kl_loss(output[mask], logit_t[mask], self.opt.kd_T)
        
        if self.maximize:
            loss = -loss

        return loss

    def unlearn_nc(self, dataset, train_mask, forget_mask):
        
        self.maximize=False
        while self.curr_step < self.opt.unlearn_iters:
            if self.curr_step < self.opt.msteps:
                self.maximize=True
                print("Gradient Ascent Step: ", self.curr_step)
                self.train_one_epoch(data=dataset, mask=forget_mask)

            self.maximize=False
            print("Gradient Descent Step: ", self.curr_step)
            self.train_one_epoch(data=dataset, mask=train_mask)
        return
    
    # scrub for label flipping
    def unlearn_nc_lf(self):
        forget_mask = self.poisoned_dataset.df_mask
        self.maximize=False
        while self.curr_step < self.opt.unlearn_iters:
            if self.curr_step < self.opt.msteps:
                self.maximize=True
                print("Gradient Ascent Step: ", self.curr_step)
                self.train_one_epoch(data=self.poisoned_dataset, mask=forget_mask)

            self.maximize=False
            print("Gradient Descent Step: ", self.curr_step)
            self.train_one_epoch(data=self.poisoned_dataset, mask=self.poisoned_dataset.dr_mask)
            train_acc, msc_rate, f1 = self.evaluate()
            print(f'Test Acc: {train_acc}, Misclassification: {msc_rate},  F1 Score: {f1}')
        return
    def train(self):
        self.unlearn_nc_lf()
    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.unlearn_iters)+'_'+str(self.opt.k)
        self.unlearn_file_prefix += '_'+str(self.opt.kd_T)+'_'+str(self.opt.alpha)+'_'+str(self.opt.msteps)
        return
