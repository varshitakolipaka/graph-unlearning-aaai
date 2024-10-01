# python run_hp_tune.py --dataset CS --df_size 3000 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type edge --db_name edge_main --gnn gcn --scrub --gif

# python eval_script.py --dataset CS --attack_type edge --df_size 3000 --start_seed 0 --end_seed 5 --db_name edge_main --log_name edge_logs --scrub --gif

# python run_hp_tune.py --dataset CS --df_size 3000 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type edge --db_name edge_cf_0.25 --cf 0.25 --gnn gcn --megu

# python eval_script.py --dataset CS --attack_type edge --df_size 3000 --start_seed 0 --end_seed 5 --db_name edge_cf_0.25 --cf 0.25 --log_name edge_logs --megu

# python run_hp_tune.py --dataset Amazon --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type label --db_name label_cf_0.75 --cf 0.75 --gnn gcn --yaum

# python eval_script.py --dataset Amazon --attack_type label --df_size 0.5 --start_seed 0 --end_seed 5 --db_name label_cf_0.75 --cf 0.75 --log_name label_cf_logs --yaum --megu --gnndelete --retrain --scrub --cacdc --gif --utu --contra_2 

# python run_hp_tune.py --dataset Amazon --df_size 0.5 

# python run_hp_tune.py --dataset Amazon --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type label --db_name label_cf_0.05 --cf 0.05 --gnn gcn --yaum

# python eval_script.py --dataset Amazon --attack_type label --df_size 0.5 --start_seed 0 --end_seed 5 --db_name label_cf_0.05 --cf 0.05 --log_name label_cf_logs --yaum --megu --gnndelete --retrain --scrub --cacdc --gif --utu --contra_2 

# python run_hp_tune.py --dataset Amazon --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type label --db_name label_cf_0.5 --cf 0.5 --gnn gcn --yaum

# python eval_script.py --dataset Amazon --attack_type label --df_size 0.5 --start_seed 0 --end_seed 5 --db_name label_cf_0.5 --cf 0.5 --log_name label_cf_logs --yaum --megu --gnndelete --retrain --scrub --cacdc --gif --utu --contra_2 

# python eval_script.py --dataset Cora --attack_type label --df_size 0.5 --start_seed 0 --end_seed 5 --db_name label_cf_0.75 --cf 0.75 --log_name label_cf_logs --megu --gnndelete --retrain --scrub --cacdc --gif --utu --contra_2 

# python run_hp_tune.py --dataset Amazon --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data3 --attack_type label --db_name label_main --gnn gcn --contra_2 --megu --scrub --retrain

# python eval_script.py --dataset CS --attack_type label --df_size 0.5 --start_seed 0 --end_seed 5 --megu --gnndelete --retrain --scrub --cacdc --yaum --gif --utu --contra_2 --db_name label_main

# python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type label --db_name label_gat_main --gnn gat --cacdc --scrub --yaum --contra_2 --megu --utu --gif --retrain --gnndelete

# python eval_script.py --gnn gat --dataset Cora --attack_type label --db_name label_gat_main --log_name label_gat_logs --start_seed 0 --end_seed 5 --cacdc --scrub --yaum --contra_2 --megu --utu --gif --retrain --gnndelete

python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type label --db_name label_gat_cf_0.05 --cf 0.05 --gnn gat --cacdc --scrub --yaum --contra_2 --megu --utu --gif --retrain --gnndelete 

python eval_script.py --gnn gat --dataset Cora --attack_type label --db_name label_gat_cf_0.05 --log_name label_gat_logs --cf 0.05 --start_seed 0 --end_seed 5 --cacdc --scrub --yaum --contra_2 --megu --utu --gif --retrain --gnndelete

python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type label --db_name label_gat_cf_0.5 --cf 0.5 --gnn gat --cacdc --scrub --yaum --contra_2 --megu --utu --gif --retrain --gnndelete 

python eval_script.py --gnn gat --dataset Cora --attack_type label --db_name label_gat_cf_0.5 --log_name label_gat_logs --cf 0.5 --start_seed 0 --end_seed 5 --cacdc --scrub --yaum --contra_2 --megu --utu --gif --retrain --gnndelete

python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type label --db_name label_gat_cf_0.75 --cf 0.75 --gnn gat --cacdc --scrub --yaum --contra_2 --megu --utu --gif --retrain --gnndelete 

python eval_script.py --gnn gat --dataset Cora --attack_type label --db_name label_gat_cf_0.75 --log_name label_gat_logs --cf 0.75 --start_seed 0 --end_seed 5 --cacdc --scrub --yaum --contra_2 --megu --utu --gif --retrain --gnndelete








# python run_hp_tune.py --dataset Amazon --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name label_cf_linked_0.05 --cf 0.05 --gnn gcn --cacdc --scrub --yaum

# python eval_script.py --dataset Amazon --attack_type label --cf 0.05 --start_seed 0 --end_seed 5 --scrub --cacdc --yaum

# python run_hp_tune.py --dataset Amazon --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name label_cf_linked_0.25 --cf 0.25 --gnn gcn --cacdc --scrub --yaum

# python eval_script.py --dataset Amazon --attack_type label --cf 0.25 --start_seed 0 --end_seed 5 --scrub --cacdc --yaum

# python run_hp_tune.py --dataset Amazon --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name label_cf_linked_0.5 --cf 0.5 --gnn gcn --cacdc --scrub --yaum



# python eval_script.py --dataset Amazon --attack_type label --cf 0.5 --start_seed 0 --end_seed 5 --scrub --cacdc --yaum

# python run_hp_tune.py --dataset Amazon --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name label_cf_linked_0.75 --cf 0.75 --gnn gcn --cacdc --scrub --yaum

# python eval_script.py --dataset Amazon --attack_type label --cf 0.75 --start_seed 0 --end_seed 5 --scrub --cacdc --yaum

# # python run_hp_tune.py --dataset Amazon --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name label_cf_linked_0.0 --cf 0.0 --gnn gcn --cacdc --scrub --yaum

# # python eval_script.py --dataset Amazon --attack_type label --cf 0.0 --start_seed 0 --end_seed 5 --scrub --cacdc --yaum

# python run_hp_tune.py --dataset Amazon --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name label_main_linked --gnn gcn --cacdc --scrub --yaum

# python eval_script.py --dataset Amazon --attack_type label --start_seed 0 --end_seed 5 --scrub --cacdc --yaum


# python run_hp_tune.py --dataset CS --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name label_cf_linked_0.05 --cf 0.05 --gnn gcn --cacdc --scrub --yaum

# python eval_script.py --dataset CS --attack_type label --cf 0.05 --start_seed 0 --end_seed 5 --scrub --cacdc --yaum

# python run_hp_tune.py --dataset CS --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name label_cf_linked_0.25 --cf 0.25 --gnn gcn --cacdc --scrub --yaum

# python eval_script.py --dataset CS --attack_type label --cf 0.25 --start_seed 0 --end_seed 5 --scrub --cacdc --yaum

# python run_hp_tune.py --dataset CS --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name label_cf_linked_0.5 --cf 0.5 --gnn gcn --cacdc --scrub --yaum

# python eval_script.py --dataset CS --attack_type label --cf 0.5 --start_seed 0 --end_seed 5 --scrub --cacdc --yaum

# python run_hp_tune.py --dataset CS --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name label_cf_linked_0.75 --cf 0.75 --gnn gcn --cacdc --scrub --yaum

# python eval_script.py --dataset CS --attack_type label --cf 0.75 --start_seed 0 --end_seed 5 --scrub --cacdc --yaum

# # python run_hp_tune.py --dataset CS --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name label_cf_0.0 --cf 0.0 --gnn gcn --cacdc --scrub --yaum

# # python eval_script.py --dataset CS --attack_type label --cf 0.0 --start_seed 0 --end_seed 5 --scrub --cacdc --yaum

# python run_hp_tune.py --dataset CS --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name label_main_linked --gnn gcn --cacdc --scrub --yaum

# python eval_script.py --dataset CS --attack_type label --start_seed 0 --end_seed 5 --scrub --cacdc --yaum

# python run_hp_tune.py --dataset Cora --df_size 750 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type edge --db_name hp_tune_edge_cf_0.25 --gnn gcn --cf 0.25 --cacdc --scrub --yaum

# python eval_script.py --dataset Cora --attack_type edge --cf 0.25 --start_seed 1 --end_seed 4 --scrub --cacdc --yaum

# python run_hp_tune.py --dataset Cora --df_size 750 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type edge --db_name hp_tune_edge_cf_0.5 --gnn gcn --cf 0.5 --cacdc

# python eval_script.py --dataset Cora --attack_type edge --cf 0.5 --start_seed 1 --end_seed 4 --scrub --cacdc --yaum

# python run_hp_tune.py --dataset Cora --df_size 750 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type edge --db_name hp_tune_edge_cf_0.75 --gnn gcn --cf 0.75 --cacdc --scrub --yaum

# python eval_script.py --dataset Cora --attack_type edge --cf 0.75 --start_seed 1 --end_seed 4 --scrub --cacdc --yaum

# python run_hp_tune.py --dataset Cora --df_size 750 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type edge --db_name hp_tune_edge_cf_0.05 --gnn gcn --cf 0.05 --cacdc --scrub --yaum

# python eval_script.py --dataset Cora --attack_type edge --cf 0.05 --start_seed 1 --end_seed 4 --scrub --cacdc --yaum

# python run_hp_tune.py --dataset Cora --df_size 750 --random_seed 0 --data_dir /scratch/akshit.sinha/data --attack_type edge --db_name hp_tune_edge --gnn gcn --cacdc

# python eval_script.py --dataset Cora --attack_type edge --start_seed 1 --end_seed 4 --scrub --cacdc --yaum

# python run_hp_tune.py --dataset CS --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name final_hp_val_acc --gnn gcn --contrastive --gif --retrain

# python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name final_hp_val_acc --gnn gcn --contrastive --gif --retrain

# python run_hp_tune.py --dataset Amazon --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label --db_name final_hp_val_acc --gnn gcn --contrastive --gif --retrain

# python run_hp_tune.py --attack edge --db_name edge_val_acc --data_dir /scratch/akshit.sinha/data --dataset Amazon --random_seed 1 --df_size 10000  --gnn gcn --scrub
# python run_hp_tune.py --attack edge --db_name edge_val_acc --data_dir /scratch/akshit.sinha/data --dataset Cora --random_seed 1 --df_size 750  --gnn gcn --scrub
# python run_hp_tune.py --attack edge --db_name edge_val_acc --data_dir /scratch/akshit.sinha/data --dataset CS --random_seed 1 --df_size 3000  --gnn gcn --scrub