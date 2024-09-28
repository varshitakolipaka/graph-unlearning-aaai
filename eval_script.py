import os
import argparse

def get_script(dataset, unlearning_model, attack, seed, cf=1.0, df_size=0.5, db_name=None):    
    dataset_to_df = {
        'Amazon': 10000,
        'Cora': 750,
        'CS': 3000,
    }
    
    cf_str = ""
    if cf < 1.0:
        cf_str = f"--corrective_frac {cf}"
    
    if attack == 'label':
        return f"python main.py --df_size 0.5 --dataset {dataset} --unlearning_model {unlearning_model} --attack_type label --random_seed {seed} --gnn gcn  --data_dir /scratch/akshit.sinha/data2 {cf_str} --db_name {db_name}"   
    if attack == 'trigger':
        return f"python main.py --df_size 0.5 --dataset {dataset} --unlearning_model {unlearning_model} --attack_type trigger --random_seed {seed} --gnn gcn  --data_dir /scratch/akshit.sinha/data2 {cf_str} --db_name {db_name}"
    
    if attack == 'random':
        return f"python main.py --df_size {df_size} --dataset {dataset} --unlearning_model {unlearning_model} --attack_type random --random_seed {seed} --gnn gcn  --data_dir /scratch/akshit.sinha/data2 {cf_str} --db_name {db_name}"

    if attack == 'edge':
        return f"python main.py --df_size {dataset_to_df[dataset]} --dataset {dataset} --unlearning_model {unlearning_model} --attack_type edge --request edge --random_seed {seed} --data_dir /scratch/akshit.sinha/data2 {cf_str} --db_name {db_name}"
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run HP tuning for various unlearning models")
    parser.add_argument('--df_size', type=float, help='Size of the dataset fraction', default=0.5)
    parser.add_argument('--attack_type', type=str, help='Type of attack to run', default='label')
    parser.add_argument('--cf', type=float, help='Corrective fraction', default=1.0)
    parser.add_argument('--dataset', type=str, help='Dataset to run the attack on', default='Cora')
    parser.add_argument('--start_seed', type=int, help='Starting seed for the attack', default=0)
    parser.add_argument('--end_seed', type=int, help='Ending seed for the attack', default=10)
    parser.add_argument('--db_name', type=str, default="hptuning", help='Database name')
    
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
    parser.add_argument('--scrub_no_kl', action='store_true', help='Run HP tuning for yaum model')
    parser.add_argument('--scrub_no_kl_combined', action='store_true', help='Run HP tuning for yaum model')

    args = parser.parse_args()
    
    unlearning_models = []
    if args.utu:
        unlearning_models.append('utu')
    if args.retrain:
        unlearning_models.append('retrain')
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

    attacks = [args.attack_type]
    datasets = [args.dataset]
    # datasets = ['Amazon']
    cfs = [args.cf]
    for dataset in datasets:
        for seed in range(args.start_seed, args.end_seed):
            for unlearning_model in unlearning_models:
                for attack in attacks:
                    for cf in cfs:
                        script = get_script(dataset, unlearning_model, attack, seed, cf, df_size=args.df_size, db_name=args.db_name)
                        # print(script)
                        os.system(script)