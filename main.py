from collections import defaultdict
import copy
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

args = parse_args()

logger = Logger(args, f"run_logs_{args.attack_type}.json")
logger.log_arguments(args)

utils.seed_everything(args.random_seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class_dataset_dict = {
    "Cora": {
        "class1": 57,
        "class2": 33,
    },
    "PubMed": {
        "class1": 2,
        "class2": 1,
    },
    "Amazon": {
        "class1": 6,
        "class2": 1,
    },
}


def train(load=False):
    if load:
        clean_data = utils.get_original_data(args.dataset)
        utils.train_test_split(clean_data, args.random_seed, args.train_ratio)
        utils.prints_stats(clean_data)

        clean_model = torch.load(
            f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_clean_model.pt"
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
            logger.log_result(
                args.random_seed, "original", {"forget": forg, "utility": util}
            )

        return clean_data

    # dataset
    print("==TRAINING==")
    clean_data = utils.get_original_data(args.dataset)
    utils.train_test_split(clean_data, args.random_seed, args.train_ratio)
    utils.prints_stats(clean_data)
    clean_model = GCN(clean_data.num_features, args.hidden_dim, clean_data.num_classes)

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
        logger.log_result(
            args.random_seed, "original", {"forget": forg, "utility": util}
        )

    # save the clean model
    os.makedirs(args.data_dir, exist_ok=True)
    torch.save(
        clean_model,
        f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_clean_model.pt",
    )

    return clean_data


def poison(clean_data=None):
    if clean_data is None:
        # load the poisoned data and model and indices from np file
        poisoned_data = torch.load(
            f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_data.pt"
        )
        poisoned_indices = torch.load(
            f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_indices.pt"
        )
        poisoned_model = torch.load(
            f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_model.pt"
        )

        if not hasattr(poisoned_data, "poisoned_nodes"):
            poisoned_data.poisoned_nodes = poisoned_indices

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
        # logger.log_result(
        #     args.random_seed, "poisoned", {"forget": forg, "utility": util}
        # )

        # print(poisoned_trainer.calculate_PSR())
        return poisoned_data, poisoned_indices, poisoned_model

    print("==POISONING==")
    if args.attack_type == "label":
        poisoned_data, poisoned_indices = label_flip_attack(
            clean_data, args.df_size, args.random_seed
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
    elif args.attack_type == "trigger":
        poisoned_data, poisoned_indices = trigger_attack(
            clean_data,
            args.df_size,
            args.poison_tensor_size,
            args.random_seed,
            args.test_poison_fraction,
            target_class=57,
        )
    poisoned_data = poisoned_data.to(device)

    if "gnndelete" in args.unlearning_model:
        # poisoned_model = GCNDelete(poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes)
        poisoned_model = GCN(
            poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes
        )
    else:
        poisoned_model = GCN(
            poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes
        )

    optimizer = torch.optim.Adam(
        poisoned_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay
    )
    poisoned_trainer = Trainer(
        poisoned_model, poisoned_data, optimizer, args.training_epochs
    )
    poisoned_trainer.train()

    # save the poisoned data and model and indices to np file
    os.makedirs(args.data_dir, exist_ok=True)

    torch.save(
        poisoned_model,
        f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_model.pt",
    )

    torch.save(
        poisoned_data,
        f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_data.pt",
    )
    torch.save(
        poisoned_indices,
        f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_indices.pt",
    )

    forg, util = poisoned_trainer.get_score(
        args.attack_type,
        class1=class_dataset_dict[args.dataset]["class1"],
        class2=class_dataset_dict[args.dataset]["class2"],
    )
    print(f"==Poisoned Model==\nForget Ability: {forg}, Utility: {util}")
    logger.log_result(args.random_seed, "poisoned", {"forget": forg, "utility": util})
    # print(f"PSR: {poisoned_trainer.calculate_PSR()}")
    return poisoned_data, poisoned_indices, poisoned_model


def unlearn(poisoned_data, poisoned_indices, poisoned_model):
    print("==UNLEARNING==")
    print(args)
    utils.find_masks(
        poisoned_data, poisoned_indices, args, attack_type=args.attack_type
    )
    if "gnndelete" in args.unlearning_model:
        unlearn_model = GCNDelete(
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
        unlearn_model = GCN(
            poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes
        )
        optimizer_unlearn = utils.get_optimizer(args, unlearn_model)
        unlearn_trainer = utils.get_trainer(
            args, unlearn_model, poisoned_data, optimizer_unlearn
        )
    else:
        optimizer_unlearn = utils.get_optimizer(args, poisoned_model)
        unlearn_trainer = utils.get_trainer(
            args, poisoned_model, poisoned_data, optimizer_unlearn
        )

    _, _, time_taken = unlearn_trainer.train()
    forg, util = unlearn_trainer.get_score(
        args.attack_type,
        class1=class_dataset_dict[args.dataset]["class1"],
        class2=class_dataset_dict[args.dataset]["class2"],
    )
    print(
        f"==Unlearned Model==\nForget Ability: {forg}, Utility: {util}, Time Taken: {time_taken}"
    )
    logger.log_result(
        args.random_seed,
        args.unlearning_model,
        {"forget": forg, "utility": util, "time_taken": time_taken},
    )
    print("==UNLEARNING DONE==")


d = {
    "retrain": {},
    "gnndelete": {
        "label": {
            "Amazon": {
                "unlearn_lr": 0.001,
                "weight_decay": 0.00002,
                "unlearning_epochs": 125,
                "alpha": 0.6,
                "loss_type": "only2_all",
            },
            "Cora": {
                "unlearn_lr": 0.0011,
                "weight_decay": 0.027,
                "unlearning_epochs": 125,
                "alpha": 0.2,
                "loss_type": "only2_layerwise",
            },
            "PubMed": {
                "unlearn_lr": 0.0055,
                "weight_decay": 0.0883,
                "unlearning_epochs": 72,
                "alpha": 0.028,
                "loss_type": "both_layerwise",
            },
        },
        "edge": {
            "Amazon": {
                "unlearn_lr": 0.000026,
                "weight_decay": 0.02,
                "unlearning_epochs": 100,
                "alpha": 0.865,
                "loss_type": "only1",
            },
            "Cora": {
                "unlearn_lr": 0.0009,
                "weight_decay": 0.04,
                "unlearning_epochs": 80,
                "alpha": 0.003,
                "loss_type": "both_layerwise",
            },
            "PubMed": {
                "unlearn_lr": 0.0001,
                "weight_decay": 0.0007,
                "unlearning_epochs": 115,
                "alpha": 0.75,
                "loss_type": "only2_layerwise",
            },
        },
    },
    "gif": {
        "label": {
            "Amazon": {
                "iteration": 650,
                "scale": 4252241435.9052677,
                "damp": 0.368,
            },
            "Cora": {"iteration": 1000, "scale": 7706555780.747042, "damp": 0.264},
            "PubMed": {"iteration": 460, "scale": 748883604.0694325, "damp": 0.886},
        },
        "edge": {
            "Amazon": {
                "iteration": 650,
                "scale": 30001980712.433876,
                "damp": 0.169,
            },
            "Cora": {"iteration": 700, "scale": 1460567545.031471, "damp": 0.5},
            "PubMed": {"iteration": 350, "scale": 2728338855.556325, "damp": 0.908},
        },
    },
    "contrastive": {
        "label": {
            "Amazon": {
                "contrastive_epochs_1": 27,
                "contrastive_epochs_2": 25,
                "unlearn_lr": 0.03959,
                "weight_decay": 4e-5,
                "contrastive_margin": 6.5,
                "contrastive_lambda": 0.97,
                "contrastive_frac": 0.02,
                "k_hop": 1,
            },
            "Cora": {
                "contrastive_epochs_1": 50,
                "contrastive_epochs_2": 35,
                "unlearn_lr": 0.0002,
                "weight_decay": 0.0057,
                "contrastive_margin": 8,
                "contrastive_lambda": 0.441,
                "contrastive_frac": 0.057,
                "k_hop": 2,
            },
            "PubMed": {
                "contrastive_epochs_1": 40,
                "contrastive_epochs_2": 30,
                "unlearn_lr": 0.002,
                "weight_decay": 0.023,
                "contrastive_margin": 120,
                "contrastive_lambda": 0.65,
                "contrastive_frac": 0.01,
                "k_hop": 2,
            },
        },
        "edge": {
            "Amazon": {
                "contrastive_epochs_1": 50,
                "contrastive_epochs_2": 50,
                "unlearn_lr": 0.03,
                "weight_decay": 0.000016,
                "contrastive_margin": 180,
                "contrastive_lambda": 0.9,
                "contrastive_frac": 0.02,
                "k_hop": 1,
            },
            "Cora": {
                "contrastive_epochs_1": 30,
                "contrastive_epochs_2": 30,
                "unlearn_lr": 0.0003,
                "weight_decay": 0.000085,
                "contrastive_margin": 300,
                "contrastive_lambda": 0.3,
                "contrastive_frac": 0.2,
                "k_hop": 1,
            },
            "PubMed": {
                "contrastive_epochs_1": 11,
                "contrastive_epochs_2": 25,
                "unlearn_lr": 0.001,
                "weight_decay": 0.0013,
                "contrastive_margin": 6,
                "contrastive_lambda": 0.05,
                "contrastive_frac": 0.02,
                "k_hop": 1,
            },
        },
    },
    "utu": {},
    "scrub": {
        "label": {
            "Amazon": {
                "unlearn_iters": 460,
                "unlearn_lr": 0.002,
                "scrubAlpha": 0.0009,
                "msteps": 60,
            },
            "Cora": {
                "unlearn_iters": 360,
                "unlearn_lr": 0.015,
                "scrubAlpha": 0.001,
                "msteps": 10,
            },
            "PubMed": {
                "unlearn_iters": 500,
                "unlearn_lr": 0.0028,
                "scrubAlpha": 0.006,
                "msteps": 16,
            },
        }
    },
    "megu": {
        "label": {
            "Amazon": {
                "unlearn_lr": 0.000002,
                "unlearning_epochs": 750,
                "kappa": 0.005,
                "alpha1": 0.564,
                "alpha2": 0.388,
            },
            "Cora": {
                "unlearn_lr": 0.000002,
                "unlearning_epochs": 150,
                "kappa": 0.024,
                "alpha1": 0.862,
                "alpha2": 0.8,
            },
            "PubMed": {
                "unlearn_lr": 0.00016,
                "unlearning_epochs": 130,
                "kappa": 0.657,
                "alpha1": 0.172,
                "alpha2": 0.3,
            },
        },
        "edge": {
            "Amazon": {
                "unlearn_lr": 0.000002,
                "unlearning_epochs": 800,
                "kappa": 0.004,
                "alpha1": 0.8,
                "alpha2": 0.1,
            },
            "Cora": {
                "unlearn_lr": 0.000063,
                "unlearning_epochs": 650,
                "kappa": 0.0015,
                "alpha1": 0.8,
                "alpha2": 0.1,
            },
            "PubMed": {
                "unlearn_lr": 0.000002,
                "unlearning_epochs": 300,
                "kappa": 0.204,
                "alpha1": 0.041,
                "alpha2": 0.8,
            },
        },
    },
}

if __name__ == "__main__":
    print("\n\n\n")

    print(args.dataset, args.attack_type)
    # clean_data = train(load=True)
    poisoned_data, poisoned_indices, poisoned_model = poison()

    try:
        params = d[args.unlearning_model][args.attack_type][args.dataset]
    except:
        params = {}
        
    print(params)

    # set args
    for key, value in params.items():
        setattr(args, key, value)
    
    print(args, "\n\n\n")

    unlearn(poisoned_data, poisoned_indices, poisoned_model)
