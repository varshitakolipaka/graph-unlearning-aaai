import argparse
import os

def run_hp_tuning(df_size, random_seed, dataset, attack_type, data_dir, db_name, gnn):
    unlearning_models = ['contra_2', 'megu', 'gnndelete', 'utu', 'gif', 'retrain', 'scrub']
    
    for model in unlearning_models:
        cmd = f"python hp_tune.py --unlearning_model {model} --dataset {dataset} --df_size {df_size} --random_seed {random_seed} --data_dir {data_dir} --attack_type {attack_type} --db_name {db_name} --gnn {gnn}"
        
        print(f"Running command: {cmd}")
        os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HP tuning for various unlearning models")
    parser.add_argument('--df_size', type=float, required=True, help='Size of the dataset fraction')
    parser.add_argument('--random_seed', type=int, required=True, help='Random seed value')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--attack_type', type=str, required=True, help='Type of attack')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the data')
    parser.add_argument('--gnn', type=str, required=True, help='GNN model to use')
    parser.add_argument('--db_name', type=str, default=None, help='Database name (only for contra_2 model)')

    args = parser.parse_args()
    run_hp_tuning(args.df_size, args.random_seed, args.dataset, args.attack_type, args.data_dir, args.db_name, args.gnn)
