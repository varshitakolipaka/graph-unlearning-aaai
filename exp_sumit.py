from collections import defaultdict
import copy
import json
import os
import torch
from framework import utils
from framework.training_args import parse_args
from models.deletion import GCNDelete
from models.models import GCN
from trainers.base import Trainer
from attacks.edge_attack import edge_attack_specific_nodes
from attacks.label_flip import label_flip_attack
from attacks.feature_attack import trigger_attack
import optuna
from optuna.samplers import TPESampler
from functools import partial
from logger import Logger
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.utils import k_hop_subgraph

args = parse_args()

utils.seed_everything(args.random_seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

with open('classes_to_poison.json', 'r') as f:
    class_dataset_dict = json.load(f)

def train(load=False):
    if load:
        clean_data = utils.get_original_data(args.dataset)
        utils.train_test_split(
            clean_data, args.random_seed, args.train_ratio, args.val_ratio
        )
        utils.prints_stats(clean_data)

        clean_model = torch.load(
            f"{args.data_dir}/{args.gnn}_{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_clean_model.pt"
        )

        optimizer = torch.optim.Adam(
            clean_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay
        )

        clean_trainer = Trainer(
            clean_model, clean_data, optimizer, args.training_epochs
        )

        if args.attack_type != "trigger":
            clean_trainer.evaluate()
            forg, util = clean_trainer.get_score(
                args.attack_type,
                class1=class_dataset_dict[args.dataset]["class1"],
                class2=class_dataset_dict[args.dataset]["class2"],
            )

            print(f"==OG Model==\nForget Ability: {forg}, Utility: {util}")
        return clean_data, clean_model

    # dataset
    print("==TRAINING==")
    clean_data = utils.get_original_data(args.dataset)
    utils.train_test_split(
        clean_data, args.random_seed, args.train_ratio, args.val_ratio
    )
    utils.prints_stats(clean_data)
    clean_model = utils.get_model(
        args, clean_data.num_features, args.hidden_dim, clean_data.num_classes
    )

    optimizer = torch.optim.Adam(
        clean_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay
    )
    clean_trainer = Trainer(clean_model, clean_data, optimizer, args.training_epochs)
    clean_trainer.train()

    if args.attack_type != "trigger":
        forg, util = clean_trainer.get_score(
            args.attack_type,
            class1=class_dataset_dict[args.dataset]["class1"],
            class2=class_dataset_dict[args.dataset]["class2"],
        )

        print(f"==OG Model==\nForget Ability: {forg}, Utility: {util}")
    return clean_data, clean_model


def poison(clean_data=None):
    if clean_data is None:
        # load the poisoned data and model and indices from np file
        poisoned_data = torch.load(
            f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_data.pt"
        )
        poisoned_model = torch.load(
            f"{args.data_dir}/{args.gnn}_{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_model.pt"
        )

        if args.attack_type == "edge":
            poisoned_indices = poisoned_data.poisoned_edge_indices
        else:
            poisoned_indices = poisoned_data.poisoned_nodes

        optimizer = torch.optim.Adam(
            poisoned_model.parameters(),
            lr=args.train_lr,
            weight_decay=args.weight_decay,
        )
        poisoned_trainer = Trainer(
            poisoned_model, poisoned_data, optimizer, args.training_epochs
        )
        poisoned_trainer.evaluate()

        forg, util = poisoned_trainer.get_score(
            args.attack_type,
            class1=class_dataset_dict[args.dataset]["class1"],
            class2=class_dataset_dict[args.dataset]["class2"],
        )
        print(f"==Poisoned Model==\nForget Ability: {forg}, Utility: {util}")

        # print(poisoned_trainer.calculate_PSR())
        return poisoned_data, poisoned_indices, poisoned_model

    print("==POISONING==")
    if args.attack_type == "label":
        poisoned_data, poisoned_indices = label_flip_attack(
            clean_data, args.df_size, args.random_seed, class_dataset_dict[args.dataset]["class1"], class_dataset_dict[args.dataset]["class2"]
        )
    elif args.attack_type == "edge":
        poisoned_data, poisoned_indices = edge_attack_specific_nodes(
            clean_data, args.df_size, args.random_seed
        )
    elif args.attack_type == "random":
        poisoned_data = copy.deepcopy(clean_data)
        poisoned_indices = torch.randperm(clean_data.num_nodes)[
            : int(clean_data.num_nodes * args.df_size)
        ]
        poisoned_data.poisoned_nodes = poisoned_indices

    poisoned_data = poisoned_data.to(device)

    poisoned_model = utils.get_model(
        args, poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes
    )

    optimizer = torch.optim.Adam(
        poisoned_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay
    )
    poisoned_trainer = Trainer(
        poisoned_model, poisoned_data, optimizer, args.training_epochs
    )
    poisoned_trainer.train()

    forg, util = poisoned_trainer.get_score(
        args.attack_type,
        class1=class_dataset_dict[args.dataset]["class1"],
        class2=class_dataset_dict[args.dataset]["class2"],
    )
    print(f"==Poisoned Model==\nForget Ability: {forg}, Utility: {util}")
    # print(f"PSR: {poisoned_trainer.calculate_PSR()}")
    return poisoned_data, poisoned_indices, poisoned_model


def unlearn(poisoned_data, poisoned_indices, poisoned_model):
    print("==UNLEARNING==")
    print(args)
    utils.find_masks(
        poisoned_data, poisoned_indices, args, attack_type=args.attack_type
    )
    if "gnndelete" in args.unlearning_model:
        unlearn_model = utils.get_model(
            args,
            poisoned_data.num_features,
            args.hidden_dim,
            poisoned_data.num_classes,
            mask_1hop=poisoned_data.sdf_node_1hop_mask,
            mask_2hop=poisoned_data.sdf_node_2hop_mask,
            mask_3hop=poisoned_data.sdf_node_3hop_mask,
        )

        # copy the weights from the poisoned model
        state_dict = poisoned_model.state_dict()
        state_dict["deletion1.deletion_weight"] = (
            unlearn_model.deletion1.deletion_weight
        )
        state_dict["deletion2.deletion_weight"] = (
            unlearn_model.deletion2.deletion_weight
        )
        state_dict["deletion3.deletion_weight"] = (
            unlearn_model.deletion3.deletion_weight
        )

        # copy the weights from the poisoned model
        unlearn_model.load_state_dict(state_dict)

        optimizer_unlearn = utils.get_optimizer(args, unlearn_model)
        unlearn_trainer = utils.get_trainer(
            args, unlearn_model, poisoned_data, optimizer_unlearn
        )
    elif "retrain" in args.unlearning_model:
        unlearn_model = utils.get_model(
            args, poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes
        )
        optimizer_unlearn = utils.get_optimizer(args, unlearn_model)
        unlearn_trainer = utils.get_trainer(
            args, unlearn_model, poisoned_data, optimizer_unlearn
        )
    else:
        unlearn_model = utils.get_model(
            args, poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes
        )
        # copy the weights from the poisoned model
        unlearn_model.load_state_dict(poisoned_model.state_dict())
        optimizer_unlearn = utils.get_optimizer(args, unlearn_model)
        unlearn_trainer = utils.get_trainer(
            args, unlearn_model, poisoned_data, optimizer_unlearn
        )

    _, _, time_taken = unlearn_trainer.train()
    if args.unlearning_model == "scrub" or args.unlearning_model == "yaum":
        if args.attack_type == "edge":
            unlearn_trainer.evaluate(is_dr=True)
        else:
            unlearn_trainer.evaluate()
    else:
        unlearn_trainer.evaluate(is_dr=True)
    forg, util = unlearn_trainer.get_score(
        args.attack_type,
        class1=class_dataset_dict[args.dataset]["class1"],
        class2=class_dataset_dict[args.dataset]["class2"],
    )
    print(
        f"==Unlearned Model==\nForget Ability: {forg}, Utility: {util}, Time Taken: {time_taken}"
    )
    print("==UNLEARNING DONE==")
    return unlearn_model

#  add a function to just take the neighbour in the train set
def experiment1(clean_model, poisoned_model, k_hop):
    #create a directory named plots
    if not os.path.exists('plots'):
        os.makedirs('plots')
    clean_model.eval()
    ori_logits = clean_model(clean_data.x, clean_data.edge_index)
    poisoned_model.eval()
    poi_logits = poisoned_model(poisoned_data.x, poisoned_data.edge_index)
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(poisoned_indices, k_hop, poisoned_data.edge_index)
    # remove the poisoned nodes from the subset
    subset_filtered = torch.tensor(list(set(subset.tolist()) - set(poisoned_indices.tolist())), device=subset.device)

    df_ori_logits= ori_logits[subset_filtered]
    df_poi_logits = poi_logits[subset_filtered]
    diff_logits = df_poi_logits - df_ori_logits
    mean_diff = diff_logits.mean(dim=1)
    plt.hist(mean_diff.detach().cpu().numpy(), width=0.1, edgecolor='black', density=True, alpha=0.7, stacked=True, label=f'{k_hop}-hop Neighborhood')

    # take the complement of subset
    num_nodes = poisoned_data.num_nodes  # Total number of nodes in the graph
    all_nodes = torch.arange(num_nodes, device=subset.device)
    non_3hop_neigh_nodes = torch.tensor(list(set(all_nodes.tolist()) - set(subset.tolist())), device=subset.device)
    complement_subset_filtered = torch.tensor(
        list(set(non_3hop_neigh_nodes.tolist()) - set(poisoned_indices.tolist())), device=subset.device
    )
    dr_ori_logits = ori_logits[complement_subset_filtered]
    dr_poi_logits = poi_logits[complement_subset_filtered]
    diff_logits = dr_poi_logits - dr_ori_logits
    mean_diff = diff_logits.mean(dim=1)
    plt.hist(mean_diff.detach().cpu().numpy(), width=0.1, edgecolor='black', alpha = 0.5, density=True, stacked=True, label='Rest of the graph')
    plt.xlabel('Mean Difference in Logits')
    plt.ylabel('Normalized Frequency')
    plt.title('Histogram of Mean Difference in Logits (Poisoned - Original)')
    plt.xlim(-2, 2)
    plt.legend()
    plt.savefig(f'plots/{k_hop}-hop neighbour DR+DF - Original vs Poisoned Logits.png')
    plt.show()
    plt.close()

    print(subset_filtered.size(), complement_subset_filtered.size())

def experiment2(poisoned_data, poisoned_indices, poisoned_model):
    models = ["gnndelete", "gif", "utu", "contrastive", "retrain", "scrub", "megu", "yaum", 'contrascent', 'cacdc']

    test_poisoned = []
    for nodes in range(0, poisoned_data.num_nodes):
        if(poisoned_data.test_mask[nodes] == 1 and (poisoned_data.y[nodes] == 5 or poisoned_data.y[nodes] == 63)):
            test_poisoned.append(nodes)

    # load df_logits if available
    if os.path.exists("unlearnt_logits.pt"):
        unlearnt_logits = torch.load("unlearnt_logits.pt")
    else:
        unlearnt_logits = []
        for i, model in enumerate(models):
            args.unlearning_model = model
            unlearnt_model = unlearn(poisoned_data, poisoned_indices, poisoned_model)
            unlearnt_model.eval()
            if(i != 5 and i != 7 and i != 8):
                logits = unlearnt_model(poisoned_data.x, poisoned_data.edge_index[:, poisoned_data.dr_mask])
            else:
                logits = unlearnt_model(poisoned_data.x, poisoned_data.edge_index)
            unlearnt_logits.append(logits)

        unlearnt_logits = torch.stack(unlearnt_logits)
        torch.save(unlearnt_logits, f"unlearnt_logits.pt")

    # Get the index of the retrain model
    retrain_index = models.index("retrain")
    retrain_logits = unlearnt_logits[retrain_index]
    df_retrain_logits = retrain_logits[poisoned_indices]
    cacdc_index = models.index("cacdc")
    cacdc_logits = unlearnt_logits[cacdc_index]
    df_cacdc_logits = cacdc_logits[poisoned_indices]
    # get abs diff between cscdc and retrain
    diff_logits = torch.abs(df_cacdc_logits - df_retrain_logits)
    mean_diff = diff_logits.mean(dim=1)
    
    for model in models:
        if (model == "cacdc" or model == "retrain"):
            continue
        plt.hist(mean_diff.detach().cpu().numpy(), width=0.3, edgecolor='black', density=True, alpha=0.7, stacked=True, label='cacdc')
        index = models.index(model)
        cur_logits = unlearnt_logits[index]
        df_cur_logits = cur_logits[poisoned_indices]
        diff_logits_here = torch.abs(df_cur_logits - df_retrain_logits)
        mean_diff_here = diff_logits_here.mean(dim=1)
        plt.hist(mean_diff_here.detach().cpu().numpy(), width=0.3, edgecolor='black', alpha = 0.5, density=True, stacked=True, label=f'{model}')
        plt.xlabel('Mean Difference in Logits')
        plt.ylabel('Normalized Frequency')
        plt.title('Histogram of Mean Difference in DF Logits Abs(Unlearnt Method - Retrain)')
        plt.legend()
        plt.savefig(f'plots/df_retrain/cacdc - {model}.png')
        plt.show()
        plt.close()

    test_df_retrain_logits = retrain_logits[test_poisoned]
    test_df_cacdc_logits = cacdc_logits[test_poisoned]
    test_diff_logits = torch.abs(test_df_cacdc_logits - test_df_retrain_logits)
    test_mean_diff = test_diff_logits.mean(dim=1)

    for model in models:
        if (model == "cacdc" or model == "retrain"):
            continue
        plt.hist(test_mean_diff.detach().cpu().numpy(), width=0.3, edgecolor='black', density=True, alpha=0.7, stacked=True, label='cacdc')
        index = models.index(model)
        cur_logits = unlearnt_logits[index]
        test_df_cur_logits = cur_logits[test_poisoned]
        diff_logits_here = torch.abs(test_df_cur_logits - test_df_retrain_logits)
        mean_diff_here = diff_logits_here.mean(dim=1)
        plt.hist(mean_diff_here.detach().cpu().numpy(), width=0.3, edgecolor='black', alpha = 0.5, density=True, stacked=True, label=f'{model}')
        plt.xlabel('Mean Difference in Logits')
        plt.ylabel('Normalized Frequency')
        plt.title('Histogram of Mean Difference in Test DF Logits Abs(Unlearnt Method - Retrain)')
        plt.legend()
        plt.savefig(f'plots/test_df_retrain/cacdc - {model}.png')
        plt.show()
        plt.close()

def experiment3(poisoned_data, poisoned_indices, poisoned_model):
    models = ["gnndelete", "gif", "utu", "contrastive", "retrain", "scrub", "megu", "yaum", 'contrascent', 'cacdc']

    test_poisoned = []
    for nodes in range(0, poisoned_data.num_nodes):
        if(poisoned_data.test_mask[nodes] == 1 and (poisoned_data.y[nodes] == 5 or poisoned_data.y[nodes] == 63)):
            test_poisoned.append(nodes)

    # load df_logits if available
    if os.path.exists("unlearnt_logits.pt"):
        unlearnt_logits = torch.load("unlearnt_logits.pt")
    else:
        unlearnt_logits = []
        for i, model in enumerate(models):
            args.unlearning_model = model
            unlearnt_model = unlearn(poisoned_data, poisoned_indices, poisoned_model)
            unlearnt_model.eval()
            if(i != 5 and i != 7 and i != 8):
                logits = unlearnt_model(poisoned_data.x, poisoned_data.edge_index[:, poisoned_data.dr_mask])
            else:
                logits = unlearnt_model(poisoned_data.x, poisoned_data.edge_index)
            unlearnt_logits.append(logits)

        unlearnt_logits = torch.stack(unlearnt_logits)
        torch.save(unlearnt_logits, f"unlearnt_logits.pt")

    poisoned_model.eval()
    poi_logits = poisoned_model(poisoned_data.x, poisoned_data.edge_index)
    df_poi_logits = poi_logits[poisoned_indices]
    test_poi_logits = poi_logits[test_poisoned]

    cacdc_index = models.index("cacdc")
    cacdc_logits = unlearnt_logits[cacdc_index]
    df_cacdc_logits = cacdc_logits[poisoned_indices]
    test_cacdc_logits = cacdc_logits[test_poisoned]

    # get abs diff between cscdc and retrain
    diff_logits = torch.abs(df_cacdc_logits - df_poi_logits)
    mean_diff = diff_logits.mean(dim=1)

    for model in models:
        if (model == "cacdc"):
            continue
        plt.hist(mean_diff.detach().cpu().numpy(), width=0.5, edgecolor='black', density=True, alpha=0.7, stacked=True, label='cacdc')
        index = models.index(model)
        cur_logits = unlearnt_logits[index]
        df_cur_logits = cur_logits[poisoned_indices]
        diff_logits_here = df_cur_logits - df_poi_logits
        mean_diff_here = diff_logits_here.mean(dim=1)
        plt.hist(mean_diff_here.detach().cpu().numpy(), width=0.3, edgecolor='black', alpha = 0.5, density=True, stacked=True, label=f'{model}')
        plt.xlabel('Mean Difference in Logits')
        plt.ylabel('Normalized Frequency')
        plt.title('Histogram of Mean Difference in Test DF Logits Abs(Unlearnt Method - Poisoned)')
        plt.legend()
        plt.savefig(f'plots/df_poisoned/cacdc - {model}.png')
        plt.show()
        plt.close()

    diff_logits = torch.abs(test_cacdc_logits - test_poi_logits)
    mean_diff = diff_logits.mean(dim=1)

    for model in models:
        if (model == "cacdc"):
            continue
        plt.hist(mean_diff.detach().cpu().numpy(), width=0.3, edgecolor='black', density=True, alpha=0.7, stacked=True, label='cacdc')
        index = models.index(model)
        cur_logits = unlearnt_logits[index]
        df_cur_logits = cur_logits[test_poisoned]
        diff_logits_here = torch.abs(df_cur_logits - test_poi_logits)
        mean_diff_here = diff_logits_here.mean(dim=1)
        plt.hist(mean_diff_here.detach().cpu().numpy(), width=0.3, edgecolor='black', alpha = 0.5, density=True, stacked=True, label=f'{model}')
        plt.xlabel('Mean Difference in Logits')
        plt.ylabel('Normalized Frequency')
        plt.title('Histogram of Mean Difference in Test DF Logits Abs(Unlearnt Method - Poisoned)')
        plt.legend()
        plt.savefig(f'plots/test_df_poisoned/cacdc - {model}.png')
        plt.show()
        plt.close()

if __name__ == "__main__":
    print("\n\n\n")

    print(args.dataset, args.attack_type)
    clean_data, clean_model = train(load=True)

    poisoned_data, poisoned_indices, poisoned_model = poison()

    # load best params file
    with open("best_params.json", "r") as f:
        d = json.load(f)

    if args.corrective_frac < 1:
        poisoned_indices = utils.sample_poison_data(poisoned_data, args.corrective_frac)
        poisoned_data.poisoned_nodes = poisoned_indices

    try:
        params = d[args.unlearning_model][args.attack_type][args.dataset]
    except:
        params = {}

    # set args
    for key, value in params.items():
        setattr(args, key, value)
    
    # experiment1(clean_model, poisoned_model, k_hop = 1)
    # experiment2(poisoned_data, poisoned_indices, poisoned_model)
    experiment3(poisoned_data, poisoned_indices, poisoned_model)

    # utils.plot_embeddings(
    #     args,
    #     unlearnt_model,
    #     poisoned_data,
    #     class1=class_dataset_dict[args.dataset]["class1"],
    #     class2=class_dataset_dict[args.dataset]["class2"],
    #     is_dr=True,
    #     name=f"unlearned_{args.unlearning_model}_2",
    # )
