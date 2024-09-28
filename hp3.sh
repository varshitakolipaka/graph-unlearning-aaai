python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name label_main --gnn gcn --cacdc

python eval_script.py --dataset Cora --attack_type label --df_size 0.5 --start_seed 0 --end_seed 5 --cacdc --db_name label_main

# python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name label_cf_0.25 --cf 0.25 --gnn gcn --cacdc --scrub --yaum

# python eval_script.py --dataset Cora --attack_type label --cf 0.25 --start_seed 0 --end_seed 5 --scrub --cacdc --yaum --db_name label_cf_0.25

# python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name label_main --gnn gcn --utu

# python eval_script.py --dataset Cora --attack_type label --df_size 0.5 --start_seed 0 --end_seed 5 --utu --db_name label_main

# python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name label_cf_0.5 --cf 0.5 --gnn gcn --cacdc --scrub --yaum --db_name label_cf_0.5

# python eval_script.py --dataset Cora --attack_type label --cf 0.5 --start_seed 0 --end_seed 5 --scrub --cacdc --yaum --db_name label_cf_0.5

# python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name label_cf_0.75 --cf 0.75 --gnn gcn --cacdc --scrub --yaum --db_name label_cf_0.75

# python eval_script.py --dataset Cora --attack_type label --cf 0.75 --start_seed 0 --end_seed 5 --scrub --cacdc --yaum --db_name label_cf_0.75