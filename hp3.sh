
python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data3 --attack_type label --db_name label_main_gat --gnn gat --cacdc --contra_2 --scrub --gnndelete

python eval_script.py --gnn gat --dataset Cora --attack_type label --db_name label_main_gat --log_name label_gat_logs --start_seed 0 --end_seed 5 --cacdc --contra_2 --scrub --gnndelete