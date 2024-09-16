python run_hp_tune.py --dataset Amazon --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name retrain_eval --gnn gcn --retrain

python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name retrain_eval --gnn gcn --retrain

python run_hp_tune.py --dataset CS --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name retrain_eval --gnn gcn --retrain

python run_hp_tune.py --attack edge --db_name retrain_eval --data_dir /scratch/akshit.sinha/data --dataset Amazon --random_seed 1 --df_size 10000  --gnn gcn --retrain
python run_hp_tune.py --attack edge --db_name retrain_eval --data_dir /scratch/akshit.sinha/data --dataset Cora --random_seed 1 --df_size 750  --gnn gcn --retrain
python run_hp_tune.py --attack edge --db_name retrain_eval --data_dir /scratch/akshit.sinha/data --dataset CS --random_seed 1 --df_size 3000  --gnn gcn --retrain

# python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name hp_tuning_new_attack --gnn gcn --gnndelete --gif --megu

# python run_hp_tune.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Amazon --random_seed 1 --df_size 10000  --gnn gcn --scrub --retrain --utu --gnndelete --gif --megu --contra_2 --db_name edge_attack