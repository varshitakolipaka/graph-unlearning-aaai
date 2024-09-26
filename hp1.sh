# python run_hp_tune.py --dataset CS --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name contrascent_eval --gnn gcn --contra_2

python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name contra_descend --gnn gcn --cacdc
python run_hp_tune.py --dataset Amazon --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name contra_descend --gnn gcn --cacdc
python run_hp_tune.py --dataset CS --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name contra_descend --gnn gcn --cacdc
python eval_script.py --dataset Cora --attack_type label --cacdc
python eval_script.py --dataset Amazon --attack_type label --cacdc
python eval_script.py --dataset CS --attack_type label --cacdc
sh get_stats.sh

# python run_hp_tune.py --dataset Amazon --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name contrascent_eval --gnn gcn --contra_2

# python run_hp_tune.py --dataset CS --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name final_hp_val_acc --gnn gcn --contra_2 --utu --megu

# python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name final_hp_val_acc --gnn gcn --contra_2 --utu --megu

# python run_hp_tune.py --dataset Amazon --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name final_hp_val_acc --gnn gcn --contra_2 --utu --megu

# python run_hp_tune.py --attack edge --db_name edge_val_acc --data_dir /scratch/akshit.sinha/data --dataset Amazon --random_seed 1 --df_size 10000  --gnn gcn --contra_2 --utu --retrain
# python run_hp_tune.py --attack edge --db_name edge_val_acc --data_dir /scratch/akshit.sinha/data --dataset Cora --random_seed 1 --df_size 750  --gnn gcn --contra_2 --utu --retrain
# python run_hp_tune.py --attack edge --db_name edge_val_acc --data_dir /scratch/akshit.sinha/data --dataset CS --random_seed 1 --df_size 3000  --gnn gcn --contra_2 --utu --retrain

# python run_hp_tune.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Computers --random_seed 1  --df_size 5000 --gnn gcn --contra_2 --scrub --db_name edge_attack

# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset CS --random_seed 0 --df_size 3000  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset CS --random_seed 1 --df_size 3000  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset CS --random_seed 2 --df_size 3000  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset CS --random_seed 3 --df_size 3000  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset CS --random_seed 4 --df_size 3000  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset CS --random_seed 5 --df_size 3000  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset CS --random_seed 6 --df_size 3000  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset CS --random_seed 7 --df_size 3000  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset CS --random_seed 8 --df_size 3000  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset CS --random_seed 9 --df_size 3000  --gnn gcn

# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Cora --random_seed 0 --df_size 750  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Cora --random_seed 1 --df_size 750  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Cora --random_seed 2 --df_size 750  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Cora --random_seed 3 --df_size 750  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Cora --random_seed 4 --df_size 750  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Cora --random_seed 5 --df_size 750  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Cora --random_seed 6 --df_size 750  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Cora --random_seed 7 --df_size 750  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Cora --random_seed 8 --df_size 750  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Cora --random_seed 9 --df_size 750  --gnn gcn

# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Amazon --random_seed 0 --df_size 10000  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Amazon --random_seed 1 --df_size 10000  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Amazon --random_seed 2 --df_size 10000  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Amazon --random_seed 3 --df_size 10000  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Amazon --random_seed 4 --df_size 10000  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Amazon --random_seed 5 --df_size 10000  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Amazon --random_seed 6 --df_size 10000  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Amazon --random_seed 7 --df_size 10000  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Amazon --random_seed 8 --df_size 10000  --gnn gcn
# python poison_attack.py --attack edge --data_dir /scratch/akshit.sinha/data --dataset Amazon --random_seed 9 --df_size 10000  --gnn gcn
