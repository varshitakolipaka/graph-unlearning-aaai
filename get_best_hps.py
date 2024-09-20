import json
import optuna
from optuna.samplers import TPESampler
from framework.training_args import parse_args

with open('classes_to_poison.json', 'r') as f:
    class_dataset_dict = json.load(f)

if __name__=="__main__":
    args = parse_args()
    
    study = optuna.create_study(
            sampler=TPESampler(seed=42),
            direction="maximize",
            study_name=f"{args.gnn}_{args.dataset}_{args.attack_type}_{args.df_size}_{args.unlearning_model}_{args.random_seed}_{class_dataset_dict[args.dataset]['class1']}_{class_dataset_dict[args.dataset]['class2']}",
            load_if_exists=True,
            storage=f"sqlite:///hp_tuning/{args.db_name}.db",
        )
    
    # get best hyperparameters
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.value}")
    
    params = best_trial.params
    print(f"Best trial params: {params}")
    
    # save to file
    try:
        with open('best_params.json', 'r') as f:
            data = json.load(f)
    except:
        data = {}
        
    with open('best_params.json', 'w') as f:
        
        # create the data object if it doesn't exist
        if args.unlearning_model not in data:
            data[args.unlearning_model] = {}
        if args.attack_type not in data[args.unlearning_model]:
            data[args.unlearning_model][args.attack_type] = {}
        if args.dataset not in data[args.unlearning_model][args.attack_type]:
            data[args.unlearning_model][args.attack_type][args.dataset] = {}
        
        # update the data object for the unlearning model, attack type and dataset key
        data[args.unlearning_model][args.attack_type][args.dataset] = params
        
        json.dump(data, f, indent=4)