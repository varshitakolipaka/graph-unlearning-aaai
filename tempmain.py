from collections import defaultdict
import copy
import json
import os
import torch
from framework import utils
from framework.training_args import parse_args
from trainers.base import Trainer
from attacks.edge_attack import edge_attack_specific_nodes
from attacks.label_flip import label_flip_attack
from attacks.feature_attack import trigger_attack

args = parse_args()


utils.seed_everything(args.random_seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def poison(clean_data=None, load=True):
    if clean_data is None:
        clean_data = utils.get_original_data(args.dataset)
        utils.train_test_split(
            clean_data, args.random_seed, args.train_ratio, args.val_ratio
        )

    if args.attack_type == "label":
        poisoned_data, poisoned_indices = label_flip_attack(
            clean_data, args.df_size, args.random_seed, args.victim_class, args.target_class
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
    elif args.attack_type == "trigger":
        poisoned_data, poisoned_indices = trigger_attack(
                clean_data, args.df_size, args.random_seed, victim_class=args.victim_class, target_class=args.target_class, trigger_size=args.trigger_size,
                test_poison_fraction=0.2
        )

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

    poisoned_trainer = Trainer(
        poisoned_model, poisoned_data, None, args.training_epochs
    )
    poisoned_trainer.evaluate()

    forg, util = poisoned_trainer.get_score(
        args.attack_type,
        class1= args.victim_class,
        class2= args.target_class,
    )

    log_val = {
        "PSR": forg,
        "utility": util,
        "df_size": args.df_size,
        "seed": args.random_seed,
        "victim_class": args.victim_class,
        "target_class": args.target_class,
        "trigger_size": args.trigger_size,
        "dataset": args.dataset
    }
    log_val_str= str(log_val)

    try:
        with open("./report.txt", "a") as f:
            f.write(f"{log_val_str}\n")
    except:
        with open("./error_logs.txt", "a") as f:
            f.write(f"{log_val_str}\n")

    print(f"==Poisoned Model==\nForget Ability: {forg}, Utility: {util}")
    return poisoned_data, poisoned_indices, poisoned_model


if __name__ == "__main__":
    poisoned_data, poisoned_indices, poisoned_model = poison()

