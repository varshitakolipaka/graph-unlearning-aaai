# python run_hp_tune.py --dataset Amazon --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name contra_no_link --gnn gcn --cacdc

python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --cf 0.25 --db_name cf_hp_tuning_0.25 --gnn gcn --cacdc --yaum --scrub

# python eval_script.py --cacdc --yaum --scrub --dataset Cora --attack_type label --cf 0.25

# python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --cf 0.5  --db_name cf_hp_tuning_0.5 --gnn gcn --cacdc --yaum --scrub 

# python eval_script.py --cacdc --yaum --scrub --dataset Cora --attack_type label --cf 0.5

# python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --cf 0.75 --db_name cf_hp_tuning0.75 --gnn gcn --cacdc --yaum --scrub

# python eval_script.py --cacdc --yaum --scrub --dataset Cora --attack_type label --cf 0.75

# python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name unlinked_evaluations_new --gnn gcn --cacdc

python eval_script.py --cacdc --dataset Cora --attack_type label

sh get_stats.sh

# python run_hp_tune.py --dataset CS --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name contra_no_link --gnn gcn --cacdc

# python run_hp_tune.py --attack edge --db_name retrain_eval --data_dir /scratch/akshit.sinha/data --dataset Amazon --random_seed 1 --df_size 10000  --gnn gcn --retrain
# python run_hp_tune.py --attack edge --db_name retrain_eval --data_dir /scratch/akshit.sinha/data --dataset Cora --random_seed 1 --df_size 750  --gnn gcn --retrain
# python run_hp_tune.py --attack edge --db_name retrain_eval --data_dir /scratch/akshit.sinha/data --dataset CS --random_seed 1 --df_size 3000  --gnn gcn --retrain

# python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name hp_tuning_new_attack --gnn gcn --gnndelete --gif --megu

# python run_hp_tune.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Amazon --random_seed 1 --df_size 10000  --gnn gcn --scrub --retrain --utu --gnndelete --gif --megu --contra_2 --db_name edge_attack