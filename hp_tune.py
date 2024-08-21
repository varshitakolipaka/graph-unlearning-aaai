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


args = parse_args()
print(args)
utils.seed_everything(args.random_seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train():
    # dataset
    print("==TRAINING==")
    clean_data= utils.get_original_data(args.dataset)
    utils.train_test_split(clean_data, args.random_seed, args.train_ratio)
    utils.prints_stats(clean_data)
    clean_model = GCN(clean_data.num_features, args.hidden_dim, clean_data.num_classes)

    optimizer = torch.optim.Adam(clean_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
    clean_trainer = Trainer(clean_model, clean_data, optimizer, args.training_epochs)
    clean_trainer.train()
    a_p, a_c = clean_trainer.subset_acc(class1=57, class2=33)
    print(f'Poisoned Acc: {a_p}, Clean Acc: {a_c}')

    # save the clean model
    os.makedirs('./data', exist_ok=True)
    torch.save(clean_model, f'./data/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_clean_model.pt')

    return clean_data

def poison(clean_data=None):
    if clean_data is None:
        # load the poisoned data and model and indices from np file
        poisoned_data = torch.load(f'./data/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_data.pt')
        poisoned_indices = torch.load(f'./data/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_indices.pt')
        poisoned_model = torch.load(f'./data/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_model.pt')

        optimizer = torch.optim.Adam(poisoned_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
        poisoned_trainer = Trainer(poisoned_model, poisoned_data, optimizer, args.training_epochs)
        poisoned_trainer.evaluate()
        a_p, a_c = poisoned_trainer.subset_acc()
        print(f'Poisoned Acc: {a_p}, Clean Acc: {a_c}')
        # print(poisoned_trainer.calculate_PSR())
        return poisoned_data, poisoned_indices, poisoned_model

    print("==POISONING==")
    if args.attack_type=="label":
        poisoned_data, poisoned_indices = label_flip_attack(clean_data, args.df_size, args.random_seed)
    elif args.attack_type=="edge":
        poisoned_data, poisoned_indices = edge_attack_specific_nodes(clean_data, args.df_size, args.random_seed)
    elif args.attack_type=="random":
        poisoned_data = copy.deepcopy(clean_data)
        poisoned_indices = torch.randperm(clean_data.num_nodes)[:int(clean_data.num_nodes*args.df_size)]
    elif args.attack_type=="trigger":
        poisoned_data, poisoned_indices = trigger_attack(clean_data, args.df_size, args.random_seed, args.test_poison_fraction)
    poisoned_data= poisoned_data.to(device)

    if "gnndelete" in args.unlearning_model:
        # poisoned_model = GCNDelete(poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes)
        poisoned_model = GCN(poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes)
    else:
        poisoned_model = GCN(poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes)

    optimizer = torch.optim.Adam(poisoned_model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
    poisoned_trainer = Trainer(poisoned_model, poisoned_data, optimizer, args.training_epochs)
    poisoned_trainer.train()

    # save the poisoned data and model and indices to np file
    os.makedirs('./data', exist_ok=True)

    torch.save(poisoned_model, f'./data/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_model.pt')

    torch.save(poisoned_data, f'./data/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_data.pt')
    torch.save(poisoned_indices, f'./data/{args.dataset}_{args.attack_type}_{args.df_size}_{args.random_seed}_poisoned_indices.pt')

    a_p, a_c = poisoned_trainer.subset_acc()
    print(f'Poisoned Acc: {a_p}, Clean Acc: {a_c}')
    # print(f"PSR: {poisoned_trainer.calculate_PSR()}")
    return poisoned_data, poisoned_indices, poisoned_model



def unlearn(poisoned_data, poisoned_indices, poisoned_model):
    print("==UNLEARNING==")

    utils.find_masks(poisoned_data, poisoned_indices, attack_type=args.attack_type)
    if "gnndelete" in args.unlearning_model:
        unlearn_model = GCNDelete(poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes, mask_1hop=poisoned_data.sdf_node_1hop_mask, mask_2hop=poisoned_data.sdf_node_2hop_mask, mask_3hop=poisoned_data.sdf_node_3hop_mask)

        # copy the weights from the poisoned model
        unlearn_model.load_state_dict(poisoned_model.state_dict())

        optimizer_unlearn= utils.get_optimizer(args, unlearn_model)
        unlearn_trainer= utils.get_trainer(args, unlearn_model, poisoned_data, optimizer_unlearn)
        unlearn_trainer.train()
    elif "retrain" in args.unlearning_model:
        unlearn_model = GCN(poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes)
        optimizer_unlearn= utils.get_optimizer(args, unlearn_model)
        unlearn_trainer= utils.get_trainer(args, unlearn_model, poisoned_data, optimizer_unlearn)
        unlearn_trainer.train()
    else:
        optimizer_unlearn= utils.get_optimizer(args, poisoned_model)
        unlearn_trainer= utils.get_trainer(args, poisoned_model, poisoned_data, optimizer_unlearn)
        unlearn_trainer.train()

    print("==UNLEARNING DONE==")

hp_tuning_params_dict = {
    'gnndelete': {
        'unlearn_lr': (1e-5, 1e-1, "log"),
        'weight_decay': (1e-5, 1e-1, "log"),
        'unlearning_epochs': (10, 200, "int"),
        'alpha': (0, 1, "float"),
        'loss_type': (["both_all", "both_layerwise", "only2_layerwise", "only2_all", "only1", "only3", "only3_all", "only3_layerwise"], "categorical"),
    },
    'gnndelete_ni': {
        'unlearn_lr': (1e-5, 1e-1, "log"),
        'weight_decay': (1e-5, 1e-1, "log"),
        'unlearning_epochs': (10, 100, "int"),
        'loss_type': (["only2_layerwise", "only2_all", "only1", "only3", "only3_all", "only3_layerwise"], "categorical"),
    },
    'gif': {
        'iteration': (10, 1000, "int"),
        'scale': (1e7, 1e11, "log"),
        'damp': (0.0, 1.0, "float"),
    },
    'gradient_ascent': {
        'unlearning_epochs': (10, 2000, "int"),
        'unlearn_lr': (1e-5, 1e-1, "log"),
        'weight_decay': (1e-5, 1e-1, "log"),
    },
    'contrastive': {
        'contrastive_epochs_1': (5, 100, "int"),
        'contrastive_epochs_2': (5, 50, "int"),
        'unlearn_lr': (1e-5, 1e-1, "log"),
        'weight_decay': (1e-5, 1e-1, "log"),
        'contrastive_margin': (1, 1e3, "log"),
        'contrastive_lambda': (0.0, 0.5, "float"),
        'contrastive_frac': (0.01, 0.2, "float"),
        'k_hop': (1, 2, "int"),
    },
    'utu': {},
    'scrub': {
        'unlearn_iters': (10, 500, "int"),
        # 'kd_T': (1, 10, "float"),
        'scrubAlpha': (1e-6, 10, "log"),
        'msteps': (10, 100, "int"),
        # 'weight_decay': (1e-5, 1e-1, "log"),
    },
    'clean': {
        'train_lr': (1e-5, 1e-1, "log"),
        'weight_decay': (1e-5, 1e-1, "log"),
        'training_epochs': (500, 3000, "int"),
    },
}

def set_hp_tuning_params(trial):
    hp_tuning_params = hp_tuning_params_dict[args.unlearning_model]
    for hp, values in hp_tuning_params.items():
        if values[1] == "categorical":
            setattr(args, hp, trial.suggest_categorical(hp, values[0]))
        elif values[2] == "int":
            setattr(args, hp, trial.suggest_int(hp, values[0], values[1]))
        elif values[2] == "float":
            setattr(args, hp, trial.suggest_float(hp, values[0], values[1]))
        elif values[2] == "log":
            setattr(args, hp, trial.suggest_float(hp, values[0], values[1], log=True))

def objective(trial, model, data):
    # Define the hyperparameters to tune
    set_hp_tuning_params(trial)

    optimizer = utils.get_optimizer(args, model)
    trainer = utils.get_trainer(args, model, data, optimizer)

    train_acc, msc_rate, time_taken = trainer.train()
    if args.attack_type=="trigger":
        psr= trainer.calculate_PSR()
        print("HEYYY")
        print(psr)
        return [train_acc, psr, time_taken]

    poison_acc, clean_acc = trainer.subset_acc(class1=57, class2=33)
    # We want to minimize misclassification rate and maximize accuracy
    return [train_acc, poison_acc, clean_acc, time_taken]

def objective_clean(trial, model, data):
    # Define the hyperparameters to tune
    set_hp_tuning_params(trial)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
    trainer = Trainer(model, data, optimizer, args.training_epochs)

    train_acc, msc_rate, time_taken = trainer.train()
    if args.attack_type=="trigger":
        psr= trainer.calculate_PSR()
        return [train_acc, psr, time_taken]

    poison_acc, clean_acc = trainer.subset_acc(class1=57, class2=33)

    # We want to minimize misclassification rate and maximize accuracy
    return [train_acc, poison_acc, clean_acc, time_taken]


if __name__ == "__main__":
    # clean_data = train()
    poisoned_data, poisoned_indices, poisoned_model = poison()
    # exit(0)
    # unlearn(poisoned_data, poisoned_indices, poisoned_model)
    utils.find_masks(poisoned_data, poisoned_indices, args, attack_type=args.attack_type)

    if "gnndelete" in args.unlearning_model:
        # Create a partial function with additional arguments
        model = GCNDelete(poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes, mask_1hop=poisoned_data.sdf_node_1hop_mask, mask_2hop=poisoned_data.sdf_node_2hop_mask)

        # copy the weights from the poisoned model
        state_dict = poisoned_model.state_dict()
        state_dict['deletion1.deletion_weight'] = model.deletion1.deletion_weight
        state_dict['deletion2.deletion_weight'] = model.deletion2.deletion_weight
        state_dict['deletion3.deletion_weight'] = model.deletion3.deletion_weight

        model.load_state_dict(state_dict)
    else:
        model = poisoned_model

    objective_func = partial(objective, model=model, data=poisoned_data)

    print("==HYPERPARAMETER TUNING==")
    if args.attack_type=="trigger":
        directions=['maximize', 'minimize', 'minimize']
    else:
        directions=['maximize', 'maximize', 'maximize', 'minimize']
    # Create a study with TPE sampler
    study = optuna.create_study(
        sampler=TPESampler(),
        directions=directions,
        study_name=f"{args.dataset}_{args.attack_type}_{args.unlearning_model}_{args.random_seed}",
        load_if_exists=True,
        storage='sqlite:///final_hptune_cora_full_seed_0_edge.db',
    )

    print("==OPTIMIZING==")

    # Optimize the objective function

    # reduce trials for utu and contrastive
    if args.unlearning_model == 'utu':
        study.optimize(objective_func, n_trials=1)
    elif args.unlearning_model == 'contrastive':
        study.optimize(objective_func, n_trials=100)
    else:
        study.optimize(objective_func, n_trials=200)
