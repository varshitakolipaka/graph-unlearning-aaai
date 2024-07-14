import torch
import tqdm
import copy
import time
import numpy as np
from os import makedirs
from os.path import exists, join
from torch.nn import functional as F
from torch_geometric.utils import negative_sampling
from ..utils import *
from .base import Trainer
from sklearn.metrics import roc_auc_score, average_precision_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Naive(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.curr_step, self.best_valid_loss = 0, float('inf')
        self.best_model = None
        self.save_files = {'train_loss': [], 'val_loss': [], 'train_time_taken': 0}

    def set_model(self, model):
        self.model = model
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.args.unlearn_lr, 
            momentum=0.9, 
            weight_decay=self.args.wd
        )
        self.scheduler = LinearLR(
            self.optimizer, 
            T=self.args.unlearn_iters * 1.25, 
            warmup_epochs=self.args.unlearn_iters // 100
        )

    def train_one_epoch(self, data):
        self.model.train()
        if self.curr_step <= self.args.unlearn_iters:
            self.optimizer.zero_grad()
            loss = self.forward_pass(data)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.curr_step += 1
            self.save_files['train_loss'].append(loss.item())
            return loss.item()

class ScrubTrainer(Naive):
    def __init__(self, args):
        super().__init__(args)
        self.args.unlearn_iters = args.unlearn_iters
        self.args.eval_on_cpu = True

    def set_og_model(self, model):
        self.og_model = copy.deepcopy(model)
        self.og_model.eval()

    def forward_pass(self, data):
        pos_edge_index = data.train_pos_edge_index
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )

        z = self.model(data.x, pos_edge_index)
        logits = self.model.decode(z, pos_edge_index, neg_edge_index)
        label = get_link_labels(pos_edge_index, neg_edge_index)

        with torch.no_grad():
            z_t = self.og_model(data.x, pos_edge_index)
            logits_t = self.og_model.decode(z_t, pos_edge_index, neg_edge_index)

        loss = F.binary_cross_entropy_with_logits(logits, label)
        loss += self.args.alpha * distill_kl_loss(logits, logits_t, self.args.kd_T)

        if self.maximize:
            loss = -loss

        return loss

    def unlearn_edge_classification(self, data):
        self.maximize = False
        while self.curr_step < self.args.unlearn_iters:
            if self.curr_step < self.args.msteps:
                self.maximize = True
                time_start = time.process_time()
                print("Gradient Ascent Step: ", self.curr_step)
                self.train_one_epoch(data)
                self.save_files['train_time_taken'] += time.process_time() - time_start
                if self.curr_step % 1 == 0:
                    self.print_metrics(self.eval(self.model, data))

            self.maximize = False
            time_start = time.process_time()
            print("Gradient Descent Step: ", self.curr_step)
            self.train_one_epoch(data)
            self.save_files['train_time_taken'] += time.process_time() - time_start
            if self.curr_step % 1 == 0:
                self.print_metrics(self.eval(self.model, data))

    def get_save_prefix(self):
        self.unlearn_file_prefix = (
            f"{self.args.pretrain_file_prefix}/{self.args.deletion_size}_"
            f"{self.args.unlearn_method}_{self.args.exp_name}_"
            f"{self.args.unlearn_iters}_{self.args.k}_{self.args.kd_T}_"
            f"{self.args.alpha}_{self.args.msteps}"
        )

    @torch.no_grad()
    def eval(self, model, data, stage='val', pred_all=False):
        model.eval()
        pos_edge_index = data[f'{stage}_pos_edge_index']
        neg_edge_index = data[f'{stage}_neg_edge_index']

        if self.args.eval_on_cpu:
            model = model.to('cpu')

        mask = data.dtrain_mask if hasattr(data, 'dtrain_mask') else data.dr_mask
        z = model(data.x, data.train_pos_edge_index[:, mask])
        logits = model.decode(z, pos_edge_index, neg_edge_index).sigmoid()
        label = get_link_labels(pos_edge_index, neg_edge_index)

        # DT AUC AUP
        loss = F.binary_cross_entropy_with_logits(logits, label).cpu().item()
        dt_auc = roc_auc_score(label.cpu(), logits.cpu())
        dt_aup = average_precision_score(label.cpu(), logits.cpu())

        # DF AUC AUP
        df_logit = [] if self.args.unlearning_model in ['original'] else model.decode(z, data.directed_df_edge_index).sigmoid().tolist()

        if df_logit:
            df_auc, df_aup = [], []
            if not hasattr(self, 'df_pos_edge') or not self.df_pos_edge:
                self.df_pos_edge = [
                    torch.zeros(data.train_pos_edge_index[:, data.dr_mask].shape[1], dtype=torch.bool)
                    for _ in range(5)
                ]
                for mask in self.df_pos_edge:
                    idx = torch.randperm(data.train_pos_edge_index[:, data.dr_mask].shape[1])[:len(df_logit)]
                    mask[idx] = True

            for mask in self.df_pos_edge:
                pos_logit = model.decode(z, data.train_pos_edge_index[:, data.dr_mask][:, mask]).sigmoid().tolist()
                logit = df_logit + pos_logit
                label = [0] * len(df_logit) + [1] * len(pos_logit)
                df_auc.append(roc_auc_score(label, logit))
                df_aup.append(average_precision_score(label, logit))

            df_auc = np.mean(df_auc)
            df_aup = np.mean(df_aup)
        else:
            df_auc = df_aup = np.nan

        log = {
            f'{stage}_loss': loss,
            f'{stage}_dt_auc': dt_auc,
            f'{stage}_dt_aup': dt_aup,
            f'{stage}_df_auc': df_auc,
            f'{stage}_df_aup': df_aup,
            f'{stage}_df_logit_mean': np.mean(df_logit) if df_logit else np.nan,
            f'{stage}_df_logit_std': np.std(df_logit) if df_logit else np.nan
        }

        if self.args.eval_on_cpu:
            model = model.to(device)

        return loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, None, log

    @torch.no_grad()
    def test(self, model, data, model_retrain=None, attack_model_all=None, attack_model_sub=None, ckpt=None):
        if ckpt == 'best' and self.args.unlearning_model != 'simple':  # Load best ckpt
            ckpt = torch.load(join(self.args.checkpoint_dir, 'model_best.pt'), map_location=device)
            model.load_state_dict(ckpt['model_state'])

        loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, test_log = self.eval(model, data, 'test', pred_all=True)

        self.trainer_log['dt_loss'] = loss
        self.trainer_log['dt_auc'] = dt_auc
        self.trainer_log['dt_aup'] = dt_aup
        self.logit_all_pair = logit_all_pair
        self.trainer_log['df_auc'] = df_auc
        self.trainer_log['df_aup'] = df_aup
        self.trainer_log['auc_sum'] = dt_auc + df_auc
        self.trainer_log['aup_sum'] = dt_aup + df_aup
        self.trainer_log['auc_gap'] = abs(dt_auc - df_auc)
        self.trainer_log['aup_gap'] = abs(dt_aup - df_aup)

        return loss, dt_auc, dt_aup, df_auc, df_aup, df_logit, logit_all_pair, test_log

    def train(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        self.set_model(model)
        self.set_og_model(model)
        self.unlearn_edge_classification(data)

    def print_metrics(self, eval_results):
        loss, dt_auc, dt_aup, df_auc, df_aup, *_ = eval_results
        metrics = {
            'train_loss': self.save_files['train_loss'][-1],
            'dt_loss': loss,
            'dt_auc': dt_auc,
            'dt_aup': dt_aup,
            'df_auc': df_auc,
            'df_aup': df_aup,
            'auc_sum': dt_auc + df_auc,
            'aup_sum': dt_aup + df_aup,
            'auc_gap': abs(dt_auc - df_auc),
            'aup_gap': abs(dt_aup - df_aup)
        }
        print(f"Epoch {self.curr_step}: {metrics}")

