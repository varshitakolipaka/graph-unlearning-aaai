import argparse
import os

def run_hp_tuning(unlearning_models, df_size, random_seed, dataset, attack_type, data_dir, db_name, gnn, cf, log_name, linked=False):
    cf_str = ""
    if cf < 1.0:
        cf_str = f"--corrective_frac {cf}"
        
    link_str = ""
    if linked:
        link_str = "--linked"
    
    for model in unlearning_models:
        if attack_type == "label" or attack_type == "random" or attack_type == "trigger":
            cmd = f"python hp_tune.py --unlearning_model {model} --dataset {dataset} --df_size {df_size} --random_seed {random_seed} --data_dir {data_dir} --attack_type {attack_type} --db_name {db_name} --gnn {gnn} {cf_str} --log_name {log_name} {link_str}"
            
            print(f"Running command: {cmd}")
            os.system(cmd)
            
            print(f"Getting best HPs for {model}")
            cmd = f"python get_best_hps.py --unlearning_model {model} --dataset {dataset} --df_size {df_size} --random_seed {random_seed} --data_dir {data_dir} --attack_type {attack_type} --db_name {db_name} --gnn {gnn} {cf_str} --log_name {log_name}"
            
            os.system(cmd)
            
        elif attack_type == "edge":
            cmd = f"python hp_tune.py --unlearning_model {model} --dataset {dataset} --df_size {df_size} --random_seed {random_seed} --data_dir {data_dir} --attack_type {attack_type} --request edge --db_name {db_name} --gnn {gnn} {cf_str} --log_name {log_name} {link_str}"
            
            print(f"Running command: {cmd}")
            os.system(cmd)
            
            print(f"Getting best HPs for {model}")
            
            cmd = f"python get_best_hps.py --unlearning_model {model} --dataset {dataset} --df_size {df_size} --random_seed {random_seed} --data_dir {data_dir} --attack_type {attack_type} --request edge --db_name {db_name} --gnn {gnn} {cf_str} --log_name {log_name}"
            
            os.system(cmd)
            
    print("HP tuning completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HP tuning for various unlearning models")
    parser.add_argument('--df_size', type=float, required=True, help='Size of the dataset fraction')
    parser.add_argument('--random_seed', type=int, required=True, help='Random seed value')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--attack_type', type=str, required=True, help='Type of attack')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the data')
    parser.add_argument('--gnn', type=str, required=True, help='GNN model to use')
    parser.add_argument('--db_name', type=str, default=None, help='Database name (only for contra_2 model)')
    parser.add_argument('--cf', type=float, default=1.0, help='Corrective fraction')
    parser.add_argument('--log_name', type=str, default='default', help='Log name')
    parser.add_argument('--linked', action='store_true', help='Run HP tuning for linked model')
    
    parser.add_argument('--contra_2', action='store_true', help='Run HP tuning for contra_2 model')
    parser.add_argument('--contrastive', action='store_true', help='Run HP tuning for contra_2 model')
    parser.add_argument('--retrain', action='store_true', help='Run HP tuning for retrain model')
    parser.add_argument('--scrub', action='store_true', help='Run HP tuning for scrub model')
    parser.add_argument('--megu', action='store_true', help='Run HP tuning for megu model')
    parser.add_argument('--gnndelete', action='store_true', help='Run HP tuning for gnndelete model')
    parser.add_argument('--utu', action='store_true', help='Run HP tuning for utu model')
    parser.add_argument('--gif', action='store_true', help='Run HP tuning for gif model')
    parser.add_argument('--ssd', action='store_true', help='Run HP tuning for ssd model')
    parser.add_argument('--yaum', action='store_true', help='Run HP tuning for yaum model')
    parser.add_argument('--contrascent', action='store_true', help='Run HP tuning for yaum model')
    parser.add_argument('--cacdc', action='store_true', help='Run HP tuning for yaum model')
    parser.add_argument('--retrain_link', action='store_true', help='Run HP tuning for yaum model')
    parser.add_argument('--scrub_no_kl', action='store_true', help='Run HP tuning for yaum model')
    parser.add_argument('--scrub_no_kl_combined', action='store_true', help='Run HP tuning for yaum model')

    args = parser.parse_args()
    
    unlearning_models = []
    if args.utu:
        unlearning_models.append('utu')
    if args.retrain:
        unlearning_models.append('retrain')
    if args.retrain_link:
        unlearning_models.append('retrain_link')
    if args.scrub:
        unlearning_models.append('scrub')
    if args.contra_2:
        unlearning_models.append('contra_2')
    if args.contrastive:
        unlearning_models.append('contrastive')
    if args.megu:
        unlearning_models.append('megu')
    if args.gnndelete:
        unlearning_models.append('gnndelete')
    if args.gif:
        unlearning_models.append('gif')
    if args.ssd:
        unlearning_models.append('ssd')
    if args.yaum:
        unlearning_models.append('yaum')
    if args.contrascent:
        unlearning_models.append('contrascent')
    if args.cacdc:
        unlearning_models.append('cacdc')
    if args.scrub_no_kl:
        unlearning_models.append('scrub_no_kl')
    if args.scrub_no_kl_combined:
        unlearning_models.append('scrub_no_kl_combined')
    
    run_hp_tuning(unlearning_models, args.df_size, args.random_seed, args.dataset, args.attack_type, args.data_dir, args.db_name, args.gnn, args.cf, args.log_name, linked=args.linked)
