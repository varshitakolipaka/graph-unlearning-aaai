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

args = parse_args()


utils.seed_everything(args.random_seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

with open("classes_to_poison.json", "r") as f:
    class_dataset_dict = json.load(f)


with open("model_seeds.json") as f:
    model_seeds = json.load(f)

logger = Logger(
    args,
    f"run_logs_{args.attack_type}_{args.df_size}_{class_dataset_dict[args.dataset]['class1']}_{class_dataset_dict[args.dataset]['class2']}",
)
logger.log_arguments(args)


def train(load=False):
    if load:
        clean_data = utils.get_original_data(args.dataset)
        # utils.train_test_split(
        #     clean_data, args.random_seed, args.train_ratio, args.val_ratio
        # )
        utils.train_test_split(
            clean_data, model_seeds[args.dataset], args.train_ratio, args.val_ratio
        )
        utils.prints_stats(clean_data)

        clean_model = torch.load(
            f"{args.data_dir}/{args.gnn}_{args.dataset}_{args.attack_type}_{args.df_size}_{model_seeds[args.dataset]}_clean_model.pt"
        )

        optimizer = torch.optim.Adam(
            clean_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay
        )

        clean_trainer = Trainer(clean_model, clean_data, optimizer, args)

        if args.attack_type != "trigger":
            clean_trainer.evaluate()
            forg, util, forget_f1, util_f1 = clean_trainer.get_score(
                args.attack_type,
                class1=class_dataset_dict[args.dataset]["class1"],
                class2=class_dataset_dict[args.dataset]["class2"],
            )

            print(
                f"==OG Model==\nForg Accuracy: {forg}, Util Accuracy: {util}, Forg F1: {forget_f1}, Util F1: {util_f1}"
            )
            logger.log_result(
                args.random_seed,
                "original",
                {
                    "forget": forg,
                    "utility": util,
                    "forget_f1": forget_f1,
                    "utility_f1": util_f1,
                },
            )

        return clean_data

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
    clean_trainer = Trainer(clean_model, clean_data, optimizer, args)
    clean_trainer.train()

    if args.attack_type != "trigger":
        acc, _, _ = clean_trainer.evaluate()

        forg, util, forget_f1, util_f1 = clean_trainer.get_score(
            args.attack_type,
            class1=class_dataset_dict[args.dataset]["class1"],
            class2=class_dataset_dict[args.dataset]["class2"],
        )

        print(
            f"==OG Model==\nForg Accuracy: {forg}, Util Accuracy: {util}, Forg F1: {forget_f1}, Util F1: {util_f1}"
        )
        logger.log_result(
            args.random_seed,
            "original",
            {
                "forget": forg,
                "utility": util,
                "forget_f1": forget_f1,
                "utility_f1": util_f1,
            },
        )
        # logger.log_result(
        #     args.random_seed, "original", {"utility": acc}
        # )

    return clean_data


def poison(clean_data=None):
    if clean_data is None:
        # load the poisoned data and model and indices from np file
        poisoned_data = torch.load(
            f"{args.data_dir}/{args.dataset}_{args.attack_type}_{args.df_size}_{model_seeds[args.dataset]}_poisoned_data.pt"
        )
        poisoned_model = torch.load(
            f"{args.data_dir}/{args.gnn}_{args.dataset}_{args.attack_type}_{args.df_size}_{model_seeds[args.dataset]}_poisoned_model.pt"
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
        poisoned_trainer = Trainer(poisoned_model, poisoned_data, optimizer, args)
        poisoned_trainer.evaluate()

        forg, util, forget_f1, util_f1 = poisoned_trainer.get_score(
            args.attack_type,
            class1=class_dataset_dict[args.dataset]["class1"],
            class2=class_dataset_dict[args.dataset]["class2"],
        )

        print(
            f"==Poisoned Model==\nForg Accuracy: {forg}, Util Accuracy: {util}, Forg F1: {forget_f1}, Util F1: {util_f1}"
        )
        logger.log_result(
            args.random_seed,
            "poisoned",
            {
                "forget": forg,
                "utility": util,
                "forget_f1": forget_f1,
                "utility_f1": util_f1,
            },
        )

        # print(poisoned_trainer.calculate_PSR())
        return poisoned_data, poisoned_indices, poisoned_model

    print("==POISONING==")
    if args.attack_type == "label":
        poisoned_data, poisoned_indices = label_flip_attack(
            clean_data,
            args.df_size,
            args.random_seed,
            class_dataset_dict[args.dataset]["class1"],
            class_dataset_dict[args.dataset]["class2"],
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
            clean_data,
            args.df_size,
            args.random_seed,
            victim_class=args.victim_class,
            target_class=args.target_class,
            trigger_size=args.trigger_size,
        )

    poisoned_data = poisoned_data.to(device)

    poisoned_model = utils.get_model(
        args, poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes
    )

    optimizer = torch.optim.Adam(
        poisoned_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay
    )
    poisoned_trainer = Trainer(
        poisoned_model, poisoned_data, optimizer, args
    )
    poisoned_trainer.train()
    acc, _, _ = poisoned_trainer.evaluate()
    forg, util, forget_f1, util_f1 = poisoned_trainer.get_score(
        args.attack_type,
        class1=class_dataset_dict[args.dataset]["class1"],
        class2=class_dataset_dict[args.dataset]["class2"],
    )

    print(
        f"==Poisoned Model==\nForg Accuracy: {forg}, Util Accuracy: {util}, Forg F1: {forget_f1}, Util F1: {util_f1}"
    )
    logger.log_result(
        args.random_seed,
        "poisoned",
        {
            "forget": forg,
            "utility": util,
            "forget_f1": forget_f1,
            "utility_f1": util_f1,
        },
    )
    # logger.log_result(args.random_seed, "poisoned", {"utility": acc})
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
    
    if args.linked:
        acc, _, _ = unlearn_trainer.evaluate(is_dr=False) # REAL
    else:
        acc, _, _ = unlearn_trainer.evaluate(is_dr=True) # REAL
    # acc, _, _ = unlearn_trainer.evaluate(is_dr=False)  # TEST
    print(acc)
    forg, util, forget_f1, util_f1 = unlearn_trainer.get_score(
        args.attack_type,
        class1=class_dataset_dict[args.dataset]["class1"],
        class2=class_dataset_dict[args.dataset]["class2"],
    )

    print(
        f"==Unlearned Model==\nForg Accuracy: {forg}, Util Accuracy: {util}, Forg F1: {forget_f1}, Util F1: {util_f1}"
    )
    logger.log_result(
        args.random_seed,
        args.unlearning_model,
        {
            "forget": forg,
            "utility": util,
            "forget_f1": forget_f1,
            "utility_f1": util_f1,
            "time_taken": time_taken,
        },
    )
    print("==UNLEARNING DONE==")
    return unlearn_model


if __name__ == "__main__":
    print("\n\n\n")

    print(args.dataset, args.attack_type)
    clean_data = train(load=False)
    # clean_data = train()

    poisoned_data, poisoned_indices, poisoned_model = poison(clean_data)
    # load best params file
    with open("best_params.json", "r") as f:
        d = json.load(f)

    if args.corrective_frac < 1:
        print("==POISONING CORRECTIVE==")
        if args.attack_type == "edge":
            poisoned_indices = poisoned_data.poisoned_edge_indices
        else:
            poisoned_indices = poisoned_data.poisoned_nodes
        print(f"No. of poisoned nodes: {len(poisoned_indices)}")
        
        if args.attack_type == "edge":
            poisoned_indices, poisoned_nodes = utils.sample_poison_data_edges(poisoned_data, args.corrective_frac)
            poisoned_data.poisoned_edge_indices = poisoned_indices
            poisoned_data.poisoned_nodes = poisoned_nodes
        else:
            poisoned_indices = utils.sample_poison_data(poisoned_indices, args.corrective_frac)
            poisoned_data.poisoned_nodes = poisoned_indices
        print(f"No. of poisoned nodes after corrective: {len(poisoned_indices)}")

    try:
        params = d[args.unlearning_model][args.experiment_name]
    except:
        params = {}
    print(params)

    # set args
    for key, value in params.items():
        setattr(args, key, value)

    unlearnt_model = unlearn(poisoned_data, poisoned_indices, poisoned_model)

    # utils.plot_embeddings(
    #     args,
    #     unlearnt_model,
    #     poisoned_data,
    #     class1=class_dataset_dict[args.dataset]["class1"],
    #     class2=class_dataset_dict[args.dataset]["class2"],
    #     is_dr=True,
    #     name=f"unlearned_{args.unlearning_model}_2",
    # )
