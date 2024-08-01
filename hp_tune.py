import copy
import torch
from framework import utils
from framework.training_args import parse_args
from models.deletion import GCNDelete
from models.models import GCN
from trainers.base import Trainer
from attacks.edge_attack import edge_attack_random_nodes
from attacks.label_flip import label_flip_attack
import optuna
from optuna.samplers import TPESampler
from functools import partial


args = parse_args()
args.unlearning_model = "contrastive"
args.dataset = "Citeseer_p"
print(args)
utils.seed_everything(args.random_seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train():
    # dataset
    print("==TRAINING==")
    clean_data= utils.get_original_data(args.dataset)
    utils.train_test_split(clean_data, args.random_seed, args.train_ratio)
    utils.prints_stats(clean_data)
    if "gnndelete" in args.unlearning_model:
        clean_model = GCNDelete(clean_data.num_features, args.hidden_dim, clean_data.num_classes)
    else:
        clean_model = GCN(clean_data.num_features, args.hidden_dim, clean_data.num_classes)

    optimizer = torch.optim.Adam(clean_model.parameters(), lr=0.01, weight_decay=5e-4)
    clean_trainer = Trainer(clean_model, clean_data, optimizer, args.training_epochs)
    clean_trainer.train()
    
    return clean_data

def poison(clean_data):
    print("==POISONING==")
    if args.attack_type=="label":
        poisoned_data, poisoned_indices = label_flip_attack(clean_data, args.df_size, args.random_seed)
    elif args.attack_type=="edge":
        poisoned_data, poisoned_indices = edge_attack_random_nodes(clean_data, args.df_size, args.random_seed)
    elif args.attack_type=="random":
        poisoned_data = copy.deepcopy(clean_data)
        poisoned_indices = torch.randperm(clean_data.num_nodes)[:int(clean_data.num_nodes*args.df_size)]
    poisoned_data= poisoned_data.to(device)

    if "gnndelete" in args.unlearning_model:
        poisoned_model = GCNDelete(poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes)
    else:
        poisoned_model = GCN(poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes)
    
    optimizer = torch.optim.Adam(poisoned_model.parameters(), lr=0.01, weight_decay=5e-4)
    poisoned_trainer = Trainer(poisoned_model, poisoned_data, optimizer, args.training_epochs)
    poisoned_trainer.train()
    
    return poisoned_data, poisoned_indices, poisoned_model

def unlearn(poisoned_data, poisoned_indices, poisoned_model):
    print("==UNLEARNING==")

    utils.find_masks(poisoned_data, poisoned_indices, attack_type=args.attack_type)
    if "gnndelete" in args.unlearning_model:
        unlearn_model = GCNDelete(poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes, mask_1hop=poisoned_data.sdf_node_1hop_mask, mask_2hop=poisoned_data.sdf_node_2hop_mask)
        
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
        'unlearning_epochs': (10, 100, "int"),
        'alpha': (0, 1, "float"),
        'loss_type': (["both_all", "both_layerwise", "only2_layerwise", "only2_all", "only1"], "categorical"),
    },
    'gnndelete_ni': {
        'unlearn_lr': (1e-5, 1e-1, "log"),
        'weight_decay': (1e-5, 1e-1, "log"),
        'unlearning_epochs': (10, 100, "int"),
        'loss_type': (["both_all", "both_layerwise", "only2_layerwise", "only2_all", "only1"], "categorical"),
    },
    'gif': {
        'iteration': (10, 100, "int"),
        'scale': (1e3, 1e6, "log"),
        'damp': (0.0, 1.0, "float"),
    },
    'gradient_ascent': {
        'unlearning_epochs': (10, 100, "int"),
    },
    'contrastive': {
        'contrastive_epochs_1': (5, 30, "int"),
        'contrastive_epochs_2': (5, 30, "int"),
        'unlearn_lr': (1e-5, 1e-1, "log"),
        'weight_decay': (1e-5, 1e-1, "log"),
        'contrastive_margin': (1e1, 1e3, "log"),
        'contrastive_lambda': (0.0, 1.0, "float"),
    },
    'utu': {}
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
    
    # We want to minimize misclassification rate and maximize accuracy
    return [(1 - train_acc) + msc_rate, train_acc, msc_rate, time_taken]

        
if __name__ == "__main__":
    clean_data = train()
    poisoned_data, poisoned_indices, poisoned_model = poison(clean_data)
    # unlearn(poisoned_data, poisoned_indices, poisoned_model)
    
    
    utils.find_masks(poisoned_data, poisoned_indices, attack_type=args.attack_type)

    if "gnndelete" in args.unlearning_model:
        # Create a partial function with additional arguments
        model = GCNDelete(poisoned_data.num_features, args.hidden_dim, poisoned_data.num_classes, mask_1hop=poisoned_data.sdf_node_1hop_mask, mask_2hop=poisoned_data.sdf_node_2hop_mask)
            
        # copy the weights from the poisoned model
        model.load_state_dict(poisoned_model.state_dict())
    else:
        model = poisoned_model
    
    objective_func = partial(objective, model=model, data=poisoned_data)
    
    print("==HYPERPARAMETER TUNING==")

    # Create a study with TPE sampler
    study = optuna.create_study(
        sampler=TPESampler(),
        directions=['minimize','maximize', 'minimize', 'minimize'],
        study_name=f"{args.dataset}_{args.attack_type}_{args.unlearning_model}",
        load_if_exists=True,
        storage='sqlite:///graph_unlearning_hp_tuning.db',
    )
    
    print("==OPTIMIZING==")

    # Optimize the objective function
    
    # reduce trials for utu and contrastive
    if args.unlearning_model == 'utu':
        study.optimize(objective_func, n_trials=1)
    elif args.unlearning_model == 'contrastive':
        study.optimize(objective_func, n_trials=30)
    else:
        study.optimize(objective_func, n_trials=100)

    # Print the best trial
    best_trial = study.best_trials
    print(f'Best trial: Accuracy: {best_trial[0].values[0]}, Misclassification: {best_trial[0].values[1]}')
    
    # Best hyperparameters
    print(f'Best hyperparameters: {best_trial[0].params}')
