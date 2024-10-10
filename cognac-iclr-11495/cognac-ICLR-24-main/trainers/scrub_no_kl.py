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
        "class1": 3,
        "class2": 4,
    },
    "Cora_p": {
        "class1": 1,
        "class2": 2,
    },
}


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

class ScrubTrainer1(Trainer):
    def __init__(self, model, poisoned_dataset, optimizer, opt):
        super().__init__(model, poisoned_dataset, optimizer, opt)
        self.opt = opt
        self.opt.unlearn_iters = opt.unlearn_iters
        self.best_model = None
        self.best_val_acc = 0
        self.curr_step = 0
        self.set_model(model)
        self.poisoned_dataset = poisoned_dataset
        self.get_masks()
        self.scheduler = LinearLR(self.optimizer, T=self.opt.unlearn_iters*1.25, warmup_epochs=self.opt.unlearn_iters//100) # Spend 1% time in warmup, and stop 66% of the way through training
        self.og_model = copy.deepcopy(model)
        self.og_model.eval()
        opt.unlearn_iters = opt.unlearn_iters
        self.opt.unlearn_iters = opt.unlearn_iters

        # set to device
        self.model.to(device)
        self.og_model.to(device)
        self.poisoned_dataset.to(device)

    def set_model(self, model):
        self.model = model

    def get_masks(self):

        if self.opt.attack_type == "edge":
            df_nodes = torch.unique(self.poisoned_dataset.edge_index[:, self.poisoned_dataset.df_mask])
            node_mask = torch.zeros(self.poisoned_dataset.num_nodes, dtype=torch.bool)
            node_mask[df_nodes] = True
            node_mask = node_mask.to(device)
            self.poisoned_dataset.node_df_mask = node_mask
            print(f"FORGETTING {node_mask.sum()} NODES....")
            self.poisoned_dataset.node_dr_mask = self.poisoned_dataset.train_mask & ~node_mask
            print(f"REMEMBERING {self.poisoned_dataset.node_dr_mask.sum()} NODES....")

        if not hasattr(self.poisoned_dataset, 'val_mask'):
            print("interesting it doesnt have a val mask")
            # If val_mask doesn't exist, create it from train_mask
            train_mask = self.poisoned_dataset.train_mask
            test_mask = self.poisoned_dataset.test_mask

            # Determine the number of nodes to move to val_mask
            # val_size = int(train_mask.sum() * self.opt.val_ratio)
            val_size = int(train_mask.sum() * self.opt.val_ratio)

            # Randomly select nodes from train_mask to create val_mask
            val_indices = torch.where(train_mask)[0][torch.randperm(train_mask.sum())[:val_size]]
            val_mask = torch.zeros_like(train_mask)
            val_mask[val_indices] = True

            # Remove val nodes from train_mask
            train_mask[val_indices] = False

            # Assign the new masks to the dataset
            self.poisoned_dataset.val_mask = val_mask
            self.poisoned_dataset.train_mask = train_mask


    def train_one_epoch(self, data, mask):
        self.model.train()
        if self.curr_step <= self.opt.unlearn_iters:
            self.optimizer.zero_grad()
            loss = self.forward_pass(data, mask)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            # print(self.scheduler.get_lr())
            self.curr_step += 1
        
        self.save_best()

        return

    def forward_pass(self, data, mask):

        output = self.model(data.x, data.edge_index[:, data.dr_mask])

        with torch.no_grad():
            logit_t = self.og_model(data.x, data.edge_index)

        loss = F.cross_entropy(output[mask], data.y[mask])

        if self.maximize:
            loss = -loss

        return loss

    def unlearn_nc_lf(self):
        forget_mask = self.poisoned_dataset.node_df_mask
        print("MEOW MEH: ", forget_mask.shape)
        self.maximize=False
        start_time = time.time()
        while self.curr_step < self.opt.unlearn_iters:
            iter_start_time = time.time()
            print("UNLEARNING STEP: ", self.curr_step, end='\r')
            if self.curr_step < self.opt.msteps:
                self.maximize=True
                self.train_one_epoch(data=self.poisoned_dataset, mask=forget_mask)

            self.maximize=False
            self.train_one_epoch(data=self.poisoned_dataset, mask=self.poisoned_dataset.node_dr_mask)


            self.unlearning_time += time.time() - iter_start_time
            if self.curr_step % 10 == 0:
                cutoff = self.save_best(is_dr=True)
                if cutoff:
                    break
            
            # print(f'Test Acc: {train_acc}, Misclassification: {msc_rate},  F1 Score: {f1}')
            # print(f"==Unlearned Model==\nForget Ability: {forg}, Utility: {util}")
        end_time = time.time()
        # load best model
        self.load_best()
        train_acc, msc_rate, f1 = self.evaluate(use_val=True, is_dr=True)
        return train_acc, msc_rate, self.best_model_time - start_time

    def train(self):
        return self.unlearn_nc_lf()

    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.unlearn_iters)+'_'+str(self.opt.k)
        return