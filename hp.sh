python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data3 --attack_type label --db_name label_cf_0.25 --cf 0.25 --gnn gcn --scrub_no_kl_2

python eval_script.py --gnn gcn --dataset Cora --attack_type label --db_name label_cf_0.25 --cf 0.25 --log_name acdc_ablations --start_seed 0 --end_seed 5 --scrub_no_kl_2






# python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type trigger --db_name trigger_main --gnn gcn --cacdc

# python eval_script.py --dataset Cora --attack_type trigger --df_size 0.5 --start_seed 0 --end_seed 5 --db_name trigger_main --log_name trigger_logs --cacdc --contra_2 --yaum --gnndelete --retrain --gif --utu --scrub --megu

# python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type edge --db_name trigger_cf_0.25 --cf 0.25 --gnn gcn --cacdc --contra_2 --yaum --gnndelete --retrain --gif --utu --scrub --megu

# python eval_script.py --dataset Cora --attack_type edge --df_size 0.5 --start_seed 0 --end_seed 5 --db_name trigger_cf_0.25 --cf 0.25 --log_name trigger_logs --cacdc --contra_2 --yaum --gnndelete --retrain --gif --utu --scrub --megu

# python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type edge --db_name trigger_cf_0.05 --cf 0.05 --gnn gcn  --cacdc --contra_2 --yaum --gnndelete --retrain --gif --utu --scrub --megu

# python eval_script.py --dataset Cora --attack_type edge --df_size 0.5 --start_seed 0 --end_seed 5 --db_name trigger_cf_0.05 --cf 0.05 --log_name trigger_logs --cacdc --contra_2 --yaum --gnndelete --retrain --gif --utu --scrub --megu