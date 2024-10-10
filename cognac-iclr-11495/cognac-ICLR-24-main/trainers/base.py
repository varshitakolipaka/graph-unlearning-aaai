import json
import os
import time
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import trange
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from framework import utils

def plot_loss_vs_epochs(loss_values):
    """
    Plots a line graph of loss vs. epochs using Seaborn.

    Parameters:
    loss_values (list or numpy array): An array of loss values.
    """
    # Create an array of epoch numbers
    loss_values = torch.stack(loss_values)
    loss_values = loss_values.cpu().detach().numpy()
    epochs = np.arange(1, len(loss_values) + 1)

    # Create a DataFrame for easier plotting with Seaborn
    data = pd.DataFrame({"Epoch": epochs, "Loss": loss_values})

    # Set the style of the plot
    sns.set(style="whitegrid")

    # Create the line plot
    plt.figure(figsize=(12, 6))
    line_plot = sns.lineplot(x="Epoch", y="Loss", data=data, marker="o", color="b")

    # Add title and labels
    line_plot.set_title("Loss vs. Epochs", fontsize=16)
    line_plot.set_xlabel("Epoch", fontsize=14)
    line_plot.set_ylabel("Loss", fontsize=14)

    # Customize the tick parameters
    line_plot.tick_params(labelsize=12)

    # Show the plot
    plt.show()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

with open("classes_to_poison.json", "r") as f:
    class_dataset_dict = json.load(f)


class Trainer:
    def __init__(self, model, data, optimizer, args):
        self.model = model.to(device)
        self.data = data.to(device)
        self.optimizer = optimizer
        self.num_epochs = args.training_epochs
        self.args = args
        self.best_state_dict = None
        self.best_val_score = 0

        self.start_time = 0
        self.unlearning_time = 0
        self.best_model_time = 0

        self.TIME_THRESHOLD = 3 # 3 seconds
        
        self.is_trigg_val = False

        self.class1 = class_dataset_dict[args.dataset]["class1"]
        self.class2 = class_dataset_dict[args.dataset]["class2"]

        self.poisoned_classes = [self.class1, self.class2]
        self.clean_classes = []
        for i in range(self.data.num_classes):
            if i not in self.poisoned_classes:
                self.clean_classes.append(i)

        # create a mask for the poisoned class nodes in test set
        self.poison_test_mask_class1 = self.data.y == self.class1 & self.data.test_mask
        self.poison_test_mask_class2 = self.data.y == self.class2 & self.data.test_mask

        self.poison_test_mask = self.poison_test_mask_class1 | self.poison_test_mask_class2
        self.clean_test_mask = ~self.poison_test_mask & self.data.test_mask

    def train(self):
        losses = []
        self.data = self.data.to(device)
        st = time.time()
        for epoch in trange(self.num_epochs, desc="Epoch"):
            self.model.train()
            z = F.log_softmax(self.model(self.data.x, self.data.edge_index), dim=1)
            loss = F.nll_loss(
                z[self.data.train_mask], self.data.y[self.data.train_mask]
            )
            loss.backward()
            losses.append(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
        time_taken = time.time() - st
        train_acc, msc_rate, f1 = self.evaluate()
        # print(f'Train Acc: {train_acc}, Misclassification: {msc_rate},  F1 Score: {f1}')
        # plot_loss_vs_epochs(losses)
        return train_acc, msc_rate, time_taken

    def all_class_acc(self):
        classes = list(range(self.data.num_classes))
        true_labels = self.true.to(device)
        pred_labels = self.pred.to(device)
        accs_clean = []

        for clean_class in classes:
            clean_indices = true_labels == clean_class
            accs_clean.append(
                accuracy_score(
                    true_labels[clean_indices].cpu(), pred_labels[clean_indices].cpu()
                )
            )
        print(f"Class Accuracies: {accs_clean}")
        accs_clean = sum(accs_clean) / len(accs_clean)
        print(f"Overall Accuracy: {accs_clean}")

    def subset_acc(self, class1=None, class2=None):

        poisoned_classes = [self.class1, self.class2]

        true_labels = self.true.to(device)
        pred_labels = self.pred.to(device)

        clean_classes = []
        for i in range(self.data.num_classes):
            if i not in poisoned_classes:
                clean_classes.append(i)

        # calculate acc separately on poisoned and non-poisoned classes
        accs_poisoned = []
        accs_clean = []
        f1_poisoned = []
        f1_clean = []

        # z = F.log_softmax(self.model(self.data.x, self.data.edge_index[:, self.data.dr_mask]), dim=1)

        for poisoned_class in poisoned_classes:
            poisoned_indices = true_labels == poisoned_class
            
            if poisoned_indices.sum() == 0:
                continue
            
            accs_poisoned.append(
                accuracy_score(
                    true_labels[poisoned_indices].cpu(), pred_labels[poisoned_indices].cpu()
                )
            )

        for clean_class in clean_classes:
            clean_indices = true_labels == clean_class
            if clean_indices.sum() == 0:
                continue
            
            accs_clean.append(
                accuracy_score(
                    true_labels[clean_indices].cpu(), pred_labels[clean_indices].cpu()
                )
            )

        # take average of the accs
        accs_poisoned = sum(accs_poisoned) / len(accs_poisoned)
        accs_clean = sum(accs_clean) / len(accs_clean)
        
        # f1_poisoned = sum(f1_poisoned) / len(f1_poisoned)
        # f1_clean = sum(f1_clean) / len(f1_clean)
        f1_poisoned = 0
        f1_clean = 0

        # auc_poisoned = sum(roc_aucs_poisoned) / len(roc_aucs_poisoned)
        # auc_clean = sum(roc_aucs_clean) / len(roc_aucs_clean)

        return accs_poisoned, accs_clean, f1_poisoned, f1_clean

    def misclassification_rate(self, true_labels, pred_labels):
        if self.class1 is None or self.class2 is None:
            return 0

        true_labels = true_labels.to(device)
        pred_labels = pred_labels.to(device)
        class1_to_class2 = (
            ((true_labels == self.class1) & (pred_labels == self.class2)).sum().item()
        )
        class2_to_class1 = (
            ((true_labels == self.class2) & (pred_labels == self.class1)).sum().item()
        )
        total_class1 = (true_labels == self.class1).sum().item()
        total_class2 = (true_labels == self.class2).sum().item()
        misclassification_rate = (class1_to_class2 + class2_to_class1) / (
            total_class1 + total_class2
        )
        return misclassification_rate

    # def evaluate(self, is_dr=False):
    #     self.model.eval()

    #     with torch.no_grad():
    #         if(is_dr):
    #             z = F.log_softmax(self.model(self.data.x, self.data.edge_index[:, self.data.dr_mask]), dim=1)
    #         else:
    #             z = F.log_softmax(self.model(self.data.x, self.data.edge_index), dim=1)
    #         loss = F.nll_loss(z[self.data.test_mask], self.data.y[self.data.test_mask]).cpu().item()
    #         pred = torch.argmax(z[self.data.test_mask], dim=1).cpu()
    #         dt_acc = accuracy_score(self.data.y[self.data.test_mask].cpu(), pred)
    #         dt_f1 = f1_score(self.data.y[self.data.test_mask].cpu(), pred, average='micro')
    #         msc_rate = self.misclassification_rate(self.data.y[self.data.test_mask].cpu(), pred)
    #         # auc = roc_auc_score(self.data.y[self.data.test_mask].cpu(), F.softmax(z[self.data.test_mask], dim=1).cpu(), multi_class='ovo')

    #     # print("AUC: ",auc)

    #     self.true = self.data.y[self.data.test_mask].cpu()
    #     self.pred = pred

    #     return dt_acc, msc_rate, dt_f1

    def evaluate(self, is_dr=False, use_val=False):
        self.model.eval()

        with torch.no_grad():
            if is_dr:
                z = F.log_softmax(
                    self.model(self.data.x, self.data.edge_index[:, self.data.dr_mask]),
                    dim=1,
                )
            else:
                z = F.log_softmax(self.model(self.data.x, self.data.edge_index), dim=1)

            if use_val:
                mask = self.data.val_mask
            else:
                mask = self.data.test_mask

            loss = F.nll_loss(z[mask], self.data.y[mask]).cpu().item()
            pred = torch.argmax(z, dim=1)
            acc = accuracy_score(self.data.y[mask].cpu(), pred[mask].cpu())
            f1 = f1_score(self.data.y[mask].cpu(), pred[mask].cpu(), average="macro")
            # msc_rate = self.misclassification_rate(self.data.y[mask].cpu(), pred[mask].cpu())

        self.z = z
        self.pred = pred[mask]
        self.true = self.data.y[mask].cpu()
        
        self.is_trigg_val = use_val

        return acc, -1, f1

    def calculate_PSR(self):
        z = self.z
        if self.is_trigg_val:
            # mask = self.data.poison_val_mask
            util_mask = self.data.clean_val_mask
            poisoned_nodes = self.data.poisoned_nodes
            mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
            mask[poisoned_nodes] = True
        else:
            mask = self.data.poison_test_mask
            util_mask = self.data.clean_test_mask
        pred = torch.argmax(z[mask], dim=1).cpu()
        
        if self.is_trigg_val:
            forg = 1 - accuracy_score(self.data.y[mask].cpu(), pred)
        else:
            forg = accuracy_score(self.data.y[mask].cpu(), pred)
        
        pred_clean = torch.argmax(z[util_mask], dim=1).cpu()
        util = accuracy_score(self.data.y[util_mask].cpu(), pred_clean)
        return forg, util

    def get_score(self, attack_type, class1=None, class2=None):
        forget_ability = None
        utility = None
        if "label" in attack_type  or attack_type == "edge":
            forget_ability, utility, forget_f1, utility_f1 = self.subset_acc(class1, class2)
            return forget_ability, utility, forget_f1, utility_f1
        elif attack_type == "trigger":
            forget_ability, utility = self.calculate_PSR()
        elif attack_type == "clean_label":
            utility, _, _ = self.evaluate()
            forget_ability = self.calculate_PSR()
        elif attack_type == "random":
            utility, _, f1 = self.evaluate()
        return forget_ability, utility, 0, 0

    def get_df_outputs(self):
        self.model.eval()
        with torch.no_grad():
            out = F.log_softmax(self.model(self.data.x, self.data.edge_index), dim=1)
        out = out[self.data.poisoned_nodes]  
        pred = torch.argmax(out, dim=1)
        # get outputs of poisoned nodes
        return pred

    def get_label_change(self):
        return sum(self.og_preds != self.get_df_outputs()).item() / len(self.og_preds)

    def validate(self, is_dr=True):
        val_acc, _, _ = self.evaluate(use_val=True, is_dr=is_dr)

        # label_change_rate = self.get_label_change()
        forg, util, forget_f1, util_f1 = self.get_score(
            self.args.attack_type,
            class1=class_dataset_dict[self.args.dataset]["class1"],
            class2=class_dataset_dict[self.args.dataset]["class2"],
        )

        score = (forg + util) / 2

        return score

    def save_best(self, is_dr=True, inf=False):
        curr_time = time.time()
        score = self.validate(is_dr)
        if not inf:
            if self.unlearning_time > self.TIME_THRESHOLD:
                print(f"Model took more than {self.TIME_THRESHOLD} seconds to train. Not saving the model.")
                return True

        if score > self.best_val_score:
                self.best_model_time = self.unlearning_time
                self.best_val_score = score
                print(f"Saving best model with score: {self.best_val_score}, and time: {self.best_model_time}")
                self.best_state_dict = self.model.state_dict()
                # Assuming 'model' is your neural network
                os.makedirs("temp", exist_ok=True)
                torch.save(self.model.state_dict(), f"temp/{self.args.experiment_name}.pth")

        return False

    def load_best(self):
        try:
            self.model.load_state_dict(torch.load(f"temp/{self.args.experiment_name}.pth"))
            # delete the model state file
            os.remove(f"temp/{self.args.experiment_name}.pth")
        except:
            print("Model state file not found.")
        return self.best_val_score
