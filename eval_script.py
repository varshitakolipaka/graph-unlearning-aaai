import os

def get_script(dataset, unlearning_model, attack, seed, cf=1.0):
    
    dataset_to_df = {
        'Amazon': 10000,
        'Cora': 750,
        'CS': 3000,
    }
    
    cf_str = ""
    if cf < 1.0:
        cf_str = f"--corrective_frac {cf}"
    
    if attack == 'label':
        return f"python main.py --df_size 0.5 --dataset {dataset} --unlearning_model {unlearning_model} --attack_type label --random_seed {seed} --gnn gcn  --data_dir /scratch/akshit.sinha/data {cf_str}"
    
    if attack == 'random':
        return f"python main.py --df_size 0.05 --dataset {dataset} --unlearning_model {unlearning_model} --attack_type random --random_seed {seed} --gnn gcn  --data_dir /scratch/akshit.sinha/data {cf_str}"

    if attack == 'edge':
        return f"python main.py --df_size {dataset_to_df[dataset]} --dataset {dataset} --unlearning_model {unlearning_model} --attack_type edge --request edge --random_seed {seed} --data_dir /scratch/akshit.sinha/data {cf_str}"

# unlearning_models = ['utu', 'scrub','gnndelete','megu','gif','cacdc', 'contrascent','retrain','yaum']
unlearning_models = ['retrain_link']
# unlearning_models = ['retrain']
# attacks = ['edge', 'label']
attacks = ['label']
datasets = ['Cora']
# datasets = ['Amazon']
cfs = [1.0]
for dataset in datasets:
    for seed in range(3):
        if seed != 2:
            continue
        for unlearning_model in unlearning_models:
            for attack in attacks:
                for cf in cfs:
                    script = get_script(dataset, unlearning_model, attack, seed, cf)
                    # print(script)
                    os.system(script)