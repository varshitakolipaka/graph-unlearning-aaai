# python eval_script.py --gnn gcn --dataset Cora --attack_type label --db_name label_cf_0.25 --log_name label_cf_logs --cf 0.25 --start_seed 0 --end_seed 5 --scrub_no_kl --scrub_no_kl_combined

python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --linked --db_name label_linked --gnn gcn --gnndelete --megu

python eval_script.py --gnn gcn --dataset Cora --attack_type label --linked --db_name label_linked --log_name label_linked_logs --start_seed 0 --end_seed 5 --gnndelete --megu

python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --linked --db_name label_linked_cf_0.25 --cf 0.25 --gnn gcn --gnndelete --megu

python eval_script.py --gnn gcn --dataset Cora --attack_type label --linked --db_name label_linked_cf_0.25 --log_name label_linked_logs --cf 0.25 --start_seed 0 --end_seed 5 --gnndelete --megu


# python eval_script.py --gnn gcn --dataset Amazon --attack_type label --db_name label_main --start_seed 0 --end_seed 1 --cacdc --scrub --yaum --contra_2 --megu --utu --gif --retrain --gnndelete