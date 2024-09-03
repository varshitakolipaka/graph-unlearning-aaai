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
        utils.train_test_split(
            clean_data, args.random_seed, args.train_ratio, args.val_ratio
        )
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
    utils.train_test_split(
        clean_data, args.random_seed, args.train_ratio, args.val_ratio
    )
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
        logger.log_result(
            args.random_seed, "poisoned", {"forget": forg, "utility": util}
        )

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
        unlearn_model = GCN(
            poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes
        )
        # copy the weights from the poisoned model
        unlearn_model.load_state_dict(poisoned_model.state_dict())
        optimizer_unlearn = utils.get_optimizer(args, unlearn_model)
        unlearn_trainer = utils.get_trainer(
            args, unlearn_model, poisoned_data, optimizer_unlearn
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
    return unlearn_model


d = {
    "retrain": {},
    "gnndelete": {
        "label": {
            "Amazon": {
                "unlearn_lr": 0.005,
                "weight_decay": 0.0141,
                "unlearning_epochs": 110,
                "alpha": 0.0055,
                "loss_type": "only3",
            },
            "Cora": {
                "unlearn_lr": 0.0199,
                "weight_decay": 0.0883,
                "unlearning_epochs": 50,
                "alpha": 0.0899,
                "loss_type": "only3_all",
            },
            "PubMed": {
                "unlearn_lr": 0.00072,
                "weight_decay": 0.000019,
                "unlearning_epochs": 183,
                "alpha": 0.13588,
                "loss_type": "only3_all",
            },
        },
        "edge": {
            "Amazon": {
                "unlearn_lr": 0.0254,
                "weight_decay": 0.000025,
                "unlearning_epochs": 110,
                "alpha": 0.9329,
                "loss_type": "only3_all",
            },
            "Cora": {
                "unlearn_lr": 0.000016,
                "weight_decay": 0.000309,
                "unlearning_epochs": 50,
                "alpha": 0.1743,
                "loss_type": "both_all",
            },
            "PubMed": {
                "unlearn_lr": 0.02,
                "weight_decay": 0.002213,
                "unlearning_epochs": 172,
                "alpha": 0.164591,
                "loss_type": "both_layerwise",
            },
        },
    },
    "gif": {
        "label": {
            "Amazon": {
                "iteration": 700,
                "scale": 50825608154.821434,
                "damp": 0.92,
            },
            "Cora": {"iteration": 60, "scale": 57264906553.625755, "damp": 0.638},
            "PubMed": {"iteration": 228, "scale": 87234771604.55779, "damp": 0.77494},
        },
        "edge": {
            "Amazon": {
                "iteration": 700,
                "scale": 18469584144.482,
                "damp": 0.38,
            },
            "Cora": {"iteration": 133, "scale": 2611000768.2023053, "damp": 0.720},
            "PubMed": {"iteration": 350, "scale": 2728338855.556325, "damp": 0.80018},
        },
    },
    "contrastive": {
        "label": {
            "Amazon": {
                "contrastive_epochs_1": 30,
                "contrastive_epochs_2": 35,
                'maximise_epochs': 0,
                "unlearn_lr": 0.016411,
                "weight_decay": 0.0005263,
                "contrastive_margin": 113.10661,
                "contrastive_lambda": 0.2,
                "contrastive_frac": 0.15,
                "k_hop": 1,
            },
            "Cora": {
                "contrastive_epochs_1": 26,
                "contrastive_epochs_2": 29,
                'maximise_epochs': 0,
                "unlearn_lr": 0.0143046,
                "weight_decay": 0.000061,
                "contrastive_margin": 482.75,
                "contrastive_lambda": 0.177,
                "contrastive_frac": 0.072,
                "k_hop": 2,
            },
            "PubMed": {
                "contrastive_epochs_1": 24,
                "contrastive_epochs_2": 28,
                "unlearn_lr": 0.025,
                "weight_decay": 0.000019,
                "contrastive_margin": 10,
                "contrastive_lambda": 0.0301,
                "contrastive_frac": 0.01569,
                "k_hop": 2,
            },
        },
        "edge": {
            "Amazon": {
                "contrastive_epochs_1": 27,
                "contrastive_epochs_2": 27,
                "unlearn_lr": 0.02271,
                "weight_decay": 0.0000186,
                "contrastive_margin": 25.053,
                "contrastive_lambda": 0.3592,
                "contrastive_frac": 0.08,
                "k_hop": 1,
            },
            "Cora": {
                "contrastive_epochs_1": 8,
                "contrastive_epochs_2": 30,
                "unlearn_lr": 0.01718,
                "weight_decay": 0.00002,
                "contrastive_margin": 87.41,
                "contrastive_lambda": 0.2194,
                "contrastive_frac": 0.0756,
                "k_hop": 1,
            },
            "PubMed": {
                "contrastive_epochs_1": 20,
                "contrastive_epochs_2": 26,
                "unlearn_lr": 0.01220,
                "weight_decay": 0.0000373,
                "contrastive_margin": 465.637,
                "contrastive_lambda": 0.1763,
                "contrastive_frac": 0.0119,
                "k_hop": 1,
            },
        },
    },
    "utu": {},
    "scrub": {
        "label": {
            "Amazon": {
                "unlearn_iters": 467,
                "unlearn_lr": 0.04673,
                "scrubAlpha": 0.000004,
                "msteps": 10,
            },
            "Cora": {
                "unlearn_iters": 34,
                "unlearn_lr": 0.00228,
                "scrubAlpha": 0.015506,
                "msteps": 418,
            },
            "PubMed": {
                "unlearn_iters": 460,
                "unlearn_lr": 0.00872,
                "scrubAlpha": 0.1755,
                "msteps": 11,
            },
        }
    },
    "megu": {
        "label": {
            "Amazon": {
                "unlearn_lr": 0.000002,
                "unlearning_epochs": 645,
                "kappa": 0.0012,
                "alpha1": 0.1580,
                "alpha2": 0.8388,
            },
            "Cora": {
                "unlearn_lr": 0.000094,
                "unlearning_epochs": 926,
                "kappa": 0.0141,
                "alpha1": 0.960,
                "alpha2": 0.0217,
            },
            "PubMed": {
                "unlearn_lr": 0.00000716,
                "unlearning_epochs": 116,
                "kappa": 0.0338,
                "alpha1": 0.766,
                "alpha2": 0.620,
            },
        },
        "edge": {
            "Amazon": {
                "unlearn_lr": 00.00006,
                "unlearning_epochs": 800,
                "kappa": 0.00193,
                "alpha1": 0.08793,
                "alpha2": 0.1095,
            },
            "Cora": {
                "unlearn_lr": 0.00078,
                "unlearning_epochs": 200,
                "kappa": 0.0033,
                "alpha1": 0.622,
                "alpha2": 0.7175,
            },
            "PubMed": {
                "unlearn_lr": 0.000871,
                "unlearning_epochs": 716,
                "kappa": 0.00514,
                "alpha1": 0.5217,
                "alpha2": 0.282,
            },
        },
    },
}

if __name__ == "__main__":
    print("\n\n\n")

    print(args.dataset, args.attack_type)
    clean_data = train(load=True)
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

    unlearnt_model = unlearn(poisoned_data, poisoned_indices, poisoned_model)

    utils.plot_embeddings(
        args,
        unlearnt_model,
        poisoned_data,
        class1=class_dataset_dict[args.dataset]["class1"],
        class2=class_dataset_dict[args.dataset]["class2"],
        is_dr=True,
        name=f"unlearned {args.unlearning_model}",
    )
