# python run_hp_tune.py --dataset CS --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name final_hp_val_acc --gnn gcn --contrastive --gif --retrain

# python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name final_hp_val_acc --gnn gcn --contrastive --gif --retrain

# python run_hp_tune.py --dataset Amazon --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name final_hp_val_acc --gnn gcn --contrastive --gif --retrain

# python run_hp_tune.py --attack edge --db_name edge_val_acc --data_dir /scratch/akshit.sinha/data --dataset Amazon --random_seed 1 --df_size 10000  --gnn gcn --scrub
# python run_hp_tune.py --attack edge --db_name edge_val_acc --data_dir /scratch/akshit.sinha/data --dataset Cora --random_seed 1 --df_size 750  --gnn gcn --scrub
python run_hp_tune.py --attack edge --db_name edge_val_acc --data_dir /scratch/akshit.sinha/data --dataset CS --random_seed 1 --df_size 3000  --gnn gcn --scrub