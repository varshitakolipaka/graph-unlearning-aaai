import torch, torchmetrics, tqdm, copy, time
# from utils import LinearLR, unlearn_func, distill_kl_loss
import numpy as np
from os import makedirs
from os.path import exists
from torch.nn import functional as F
from ..utils import *
from .base import Trainer

class Naive(Trainer):
    def __init__(self, opt):
        self.opt = opt
        self.opt.unlearn_iters = opt.unlearn_iters
        self.curr_step, self.best_top1 = 0, 0
        self.best_model = None
        self.model = None
        self.save_files = {'train_top1': [], 'val_top1': [], 'train_time_taken': 0}
        self.optimizer = None
        self.scheduler = None
        self.top1 = torchmetrics.Accuracy(task="multiclass", num_classes=7)

    def set_model(self, model):
        self.model = model

    def set_optimizer_scheduler(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opt.unlearn_lr, momentum=0.9, weight_decay=self.opt.wd)
        self.scheduler = LinearLR(self.optimizer, T=self.opt.unlearn_iters * 1.25, warmup_epochs=self.opt.unlearn_iters // 100) # Spend 1% time in warmup, and stop 66% of the way through training 

    def train_one_epoch(self, data, mask):
        self.model.train()
        self.top1.reset()

        if self.curr_step <= self.opt.unlearn_iters:
            self.optimizer.zero_grad()
            loss = self.forward_pass(data, mask)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.curr_step += 1

        top1 = self.top1.compute().item()
        self.top1.reset()
        self.save_files['train_top1'].append(top1)
        return

    def eval(self, data, save_model=True, save_preds=False):
        self.model.eval()
        self.top1.reset()

        with torch.no_grad():
            output = self.model(data.x, data.edge_index)
            pred = torch.argmax(output, dim=1)
            self.top1(pred[data.test_mask], data.y[data.test_mask])

        top1 = self.top1.compute().item()
        self.top1.reset()

        if save_model:
            self.save_files['val_top1'].append(top1)
            if top1 > self.best_top1:
                self.best_top1 = top1
                self.best_model = copy.deepcopy(self.model).cpu()

        self.model.train()
        if save_preds:
            preds = np.concatenate(preds, axis=0)
            targets = np.concatenate(targets, axis=0)
            return preds, targets
        return

    def unlearn(self, train_loader, test_loader, eval_loaders=None):
        while self.curr_step < self.opt.unlearn_iters:
            time_start = time.process_time()
            self.train_one_epoch(loader=train_loader)
            self.eval(test_loader)
            self.save_files['train_time_taken'] += time.process_time() - time_start
        return

class ScrubTrainer(Naive):
    def __init__(self, opt):
        super().__init__(opt)
        self.og_model = None

    def forward_pass(self, data, mask):
        output = self.model(data.x, data.edge_index)
        with torch.no_grad():
            logit_t = self.og_model(data.x, data.edge_index)
        loss = F.cross_entropy(output[mask], data.y[mask])
        loss += self.opt.scrubAlpha * distill_kl_loss(output[mask], logit_t[mask], self.opt.kd_T)
        if self.maximize:
            loss = -loss
        pred = torch.argmax(output, dim=1)
        self.top1(pred[mask], data.y[mask])
        train_acc, val_acc, test_acc = test(self.model, data)
        print(f'Loss: {loss:.4f}, Train Acc: {train_acc:.2f}, Val Acc: {val_acc:.2f}, Test Acc: {test_acc:.2f}')
        return loss

    def train(self, model, data, optimizer, args, logits_ori=None, attack_model=None, attack_model_sub=None):
        self.set_model(model)
        self.set_optimizer_scheduler()
        self.og_model = copy.deepcopy(model)
        self.og_model.eval()
        self.unlearn_nc(dataset=data, train_mask=data.dr_mask, forget_mask=data.df_mask)

    def unlearn_nc(self, dataset, train_mask, forget_mask):
        self.maximize = False
        while self.curr_step < self.opt.unlearn_iters:
            if self.curr_step < self.opt.msteps:
                self.maximize = True
                time_start = time.process_time()
                print("Gradient Ascent Step: ", self.curr_step)
                self.train_one_epoch(data=dataset, mask=forget_mask)
                self.save_files['train_time_taken'] += time.process_time() - time_start
                self.eval(data=dataset)

            self.maximize = False
            time_start = time.process_time()
            print("Gradient Descent Step: ", self.curr_step)
            self.train_one_epoch(data=dataset, mask=train_mask)
            self.save_files['train_time_taken'] += time.process_time() - time_start
            self.eval(data=dataset)
        return

    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix + '/' + str(self.opt.deletion_size) + '_' + self.opt.unlearn_method + '_' + self.opt.exp_name
        self.unlearn_file_prefix += '_' + str(self.opt.unlearn_iters) + '_' + str(self.opt.k)
        self.unlearn_file_prefix += '_' + str(self.opt.kd_T) + '_' + str(self.opt.scrubAlpha) + '_' + str(self.opt.msteps)
        return