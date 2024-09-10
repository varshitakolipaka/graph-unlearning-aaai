import torch, torchmetrics, tqdm, copy, time
import numpy as np
from os import makedirs
from os.path import exists
from torch.nn import functional as F
from framework.training_args import parse_args
opt = parse_args()
from .base import Trainer
from torch.optim.lr_scheduler import _LRScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        "class1": 6,
        "class2": 1,
    },
    "Cora_p": {
        "class1": 1,
        "class2": 2,
    },
}

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

class GrubTrainer(Trainer):

    def __init__(self, model, poisoned_dataset, optimizer, opt):
        super().__init__(model, poisoned_dataset, optimizer)
        self.opt = opt
        self.opt.unlearn_iters = opt.unlearn_iters
        self.best_model = None
        self.best_combined_score = -float('inf')
        self.curr_step = 0
        self.set_model(model)
        self.poisoned_dataset = poisoned_dataset
        full_edge_index = self.poisoned_dataset.edge_index
        dr_mask = self.poisoned_dataset.dr_mask

        self.edge_index = full_edge_index[:, dr_mask]
        
        self.forget_mask = self.poisoned_dataset.node_df_mask
        self.retain_mask = self.poisoned_dataset.node_dr_mask
        
        # Separate optimizers for ascent and descent
        self.ascent_optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.ascent_lr)
        self.descent_optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.descent_lr)
        
        # Separate schedulers
        self.ascent_scheduler = LinearLR(self.ascent_optimizer, T=self.opt.unlearn_iters*1.25, warmup_epochs=self.opt.unlearn_iters//100)
        self.descent_scheduler = LinearLR(self.descent_optimizer, T=self.opt.unlearn_iters*1.25, warmup_epochs=self.opt.unlearn_iters//100)
        
        self.og_model = copy.deepcopy(model)
        self.og_model.eval()

        self.model.to(device)
        self.og_model.to(device)
        self.poisoned_dataset.to(device)

    def set_model(self, model):
        self.model = model

    def forward_pass(self, data, mask):

        output = self.model(data.x, self.edge_index)
        loss = F.cross_entropy(output[mask], data.y[mask])

        if self.maximize:
            loss = -loss

        return loss

    def train_one_epoch(self, data, mask):
        self.model.train()
        if self.curr_step <= self.opt.unlearn_iters:
            optimizer = self.ascent_optimizer if self.maximize else self.descent_optimizer
            scheduler = self.ascent_scheduler if self.maximize else self.descent_scheduler
            
            optimizer.zero_grad()
            loss = self.forward_pass(data, mask)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Evaluate and update best model
            train_acc, msc_rate, f1 = self.evaluate(use_val=True)
            forg, util = self.get_score(self.opt.attack_type, 
                                        class1=class_dataset_dict[self.opt.dataset]["class1"], 
                                        class2=class_dataset_dict[self.opt.dataset]["class2"])
            
            combined_score = 0.5 * forg + 0.5 * util
            if combined_score > self.best_combined_score:
                print(f"New best model found! Combined Score: {combined_score:.4f}")
                self.best_combined_score = combined_score
                with open(self.opt.unlearning_model + '_best_model.pth', 'wb') as f:
                    torch.save(self.model.state_dict(), f)
            else:
                print(f"Not updating... Combined Score: {combined_score:.4f}")
            
            self.curr_step += 1

    def unlearn_nc_lf(self):
        train_acc, msc_rate, f1 = self.evaluate(use_val=True)
        forg, util = self.get_score(self.opt.attack_type, class1=class_dataset_dict[self.opt.dataset]["class1"], class2=class_dataset_dict[self.opt.dataset]["class2"])
        print("SCORES AT THE STARTTT:", forg, util)
        self.maximize=False
        start_time = time.time()
        while self.curr_step < self.opt.unlearn_iters:
            print("UNLEARNING STEP: ", self.curr_step, end='\r')
            print("yo wtf")
            if self.curr_step < self.opt.msteps:
                self.maximize=True
                print("Gradient Ascent Step: ", self.curr_step)
                self.train_one_epoch(data=self.poisoned_dataset, mask=self.forget_mask)
                print("meow")
                print("SCORESSS:", forg, util)


            self.maximize=False
            print("Gradient Descent Step: ", self.curr_step)
            self.train_one_epoch(data=self.poisoned_dataset, mask=self.retain_mask)
            train_acc, msc_rate, f1 = self.evaluate(use_val=True)
            forg, util = self.get_score(self.opt.attack_type, class1=class_dataset_dict[self.opt.dataset]["class1"], class2=class_dataset_dict[self.opt.dataset]["class2"])
            print("SCORESSS:", forg, util)
            print("-"*50)
        end_time = time.time()
        # load best model
        with open(self.opt.unlearning_model + '_best_model.pth', 'rb') as f:
            self.model.load_state_dict(torch.load(f))
        train_acc, msc_rate, f1 = self.evaluate(use_val=True)
        return train_acc, msc_rate, end_time - start_time
    

    def train(self):
        return self.unlearn_nc_lf()

    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.unlearn_iters)+'_'+str(self.opt.k)
        self.unlearn_file_prefix += '_'+str(self.opt.kd_T)+'_'+str(self.opt.alpha)+'_'+str(self.opt.msteps)
        return