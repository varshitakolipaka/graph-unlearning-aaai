python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name label_cf_0.25 --cf 0.25 --gnn gat --cacdc --scrub --yaum --contra_2 --megu --utu --gif --retrain --gnndelete

python eval_script.py --gnn gat --dataset Cora --attack_type label --db_name label_cf_0.25 --cf 0.25 --log_name label_gat_logs --start_seed 0 --end_seed 5 --cacdc --scrub --yaum --contra_2 --megu --utu --gif --retrain --gnndelete

python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name label_cf_0.5 --cf 0.5 --gnn gat --cacdc --scrub --yaum --contra_2 --megu --utu --gif --retrain --gnndelete

python eval_script.py --gnn gat --dataset Cora --attack_type label --db_name label_cf_0.5 --cf 0.5 --log_name label_gat_logs --start_seed 0 --end_seed 5 --cacdc --scrub --yaum --contra_2 --megu --utu --gif --retrain --gnndelete

python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name label_cf_0.75 --cf 0.75 --gnn gat --cacdc --scrub --yaum --contra_2 --megu --utu --gif --retrain --gnndelete

python eval_script.py --gnn gat --dataset Cora --attack_type label --db_name label_cf_0.75 --cf 0.75 --log_name label_gat_logs --start_seed 0 --end_seed 5 --cacdc --scrub --yaum --contra_2 --megu --utu --gif --retrain --gnndelete






# python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type trigger --db_name trigger_cf_0.0 --cf 0.0 --gnn gcn --cacdc --contra_2 --yaum --scrub

# python eval_script.py --dataset Cora --attack_type trigger --df_size 0.5 --start_seed 0 --end_seed 5 --db_name trigger_cf_0.0 --cf 0.0 --log_name trigger_logs --cacdc --contra_2 --yaum --scrub

# python run_hp_tune.py --dataset CS --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type edge --db_name edge_cf_0.75 --cf 0.75 --gnn gcn --cacdc --contra_2 --yaum --gnndelete --retrain --gif --utu --scrub --megu

# python eval_script.py --dataset CS --attack_type edge --df_size 0.5 --start_seed 0 --end_seed 5 --db_name edge_cf_0.75 --cf 0.75 --log_name edge_logs --cacdc --contra_2 --yaum --gnndelete --retrain --gif --utu --scrub --megu

# python run_hp_tune.py --dataset CS --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type edge --db_name edge_cf_0.5 --cf 0.5 - --gnn gcn-cacdc --contra_2

# python eval_script.py --dataset CS --attack_type edge --df_size 0.5 --start_seed 0 --end_seed 5 --db_name edge_cf_0.5 --cf 0.5 --log_name edge_logs --cacdc --contra_2 --yaum --gnndelete --retrain --gif --utu --scrub --megu