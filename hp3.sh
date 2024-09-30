python run_hp_tune.py --dataset CS --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type label --db_name label_cf_0.25 --cf 0.25 --gnn gcn  --yaum 

python eval_script.py --dataset CS --attack_type label --df_size 0.5 --start_seed 0 --end_seed 5 --db_name label_cf_0.25 --cf 0.25 --log_name label_cf_logs --yaum

python run_hp_tune.py --dataset CS --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type label --db_name label_cf_0.75 --cf 0.75 --gnn gcn --yaum

python eval_script.py --dataset CS --attack_type label --df_size 0.5 --start_seed 0 --end_seed 5 --db_name label_cf_0.75 --cf 0.75 --log_name label_cf_logs --yaum --megu --gnndelete --retrain --scrub --cacdc --gif --utu --contra_2 

python run_hp_tune.py --dataset CS --df_size 0.5 

python run_hp_tune.py --dataset CS --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type label --db_name label_cf_0.05 --cf 0.05 --gnn gcn --yaum

python eval_script.py --dataset CS --attack_type label --df_size 0.5 --start_seed 0 --end_seed 5 --db_name label_cf_0.05 --cf 0.05 --log_name label_cf_logs --yaum --megu --gnndelete --retrain --scrub --cacdc --gif --utu --contra_2 

python run_hp_tune.py --dataset CS --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type label --db_name label_cf_0.5 --cf 0.5 --gnn gcn --yaum

python eval_script.py --dataset Amazon --attack_type label --df_size 0.5 --start_seed 0 --end_seed 5 --db_name label_cf_0.5 --cf 0.5 --log_name label_cf_logs --yaum --megu --gnndelete --retrain --scrub --cacdc --gif --utu --contra_2 