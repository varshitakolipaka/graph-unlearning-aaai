python run_hp_tune.py --dataset CS --df_size 3000 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type edge --db_name edge_cf_0.25 --cf 0.25 --gnn gcn --cacdc --contra_2 --yaum --gnndelete --retrain --gif --utu --scrub --megu

python eval_script.py --dataset CS --attack_type edge --df_size 10000 --start_seed 0 --end_seed 5 --db_name edge_cf_0.25 --cf 0.25 --log_name edge_logs --cacdc --contra_2 --yaum --gnndelete --retrain --gif --utu --scrub --megu

python run_hp_tune.py --dataset CS --df_size 3000 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type edge --db_name edge_cf_0.05 --cf 0.05 --gnn gcn  --cacdc --contra_2 --yaum --gnndelete --retrain --gif --utu --scrub --megu

python eval_script.py --dataset CS --attack_type edge --df_size 10000 --start_seed 0 --end_seed 5 --db_name edge_cf_0.05 --cf 0.05 --log_name edge_logs --cacdc --contra_2 --yaum --gnndelete --retrain --gif --utu --scrub --megu

python run_hp_tune.py --dataset CS --df_size 3000 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type edge --db_name edge_cf_0.75 --cf 0.75 --gnn gcn --cacdc --contra_2 --yaum --gnndelete --retrain --gif --utu --scrub --megu

python eval_script.py --dataset CS --attack_type edge --df_size 10000 --start_seed 0 --end_seed 5 --db_name edge_cf_0.75 --cf 0.75 --log_name edge_logs --cacdc --contra_2 --yaum --gnndelete --retrain --gif --utu --scrub --megu

python run_hp_tune.py --dataset CS --df_size 3000 --random_seed 1 --data_dir /scratch/akshit.sinha/data3 --attack_type edge --db_name edge_cf_0.5 --cf 0.5 - --gnn gcn-cacdc --contra_2

python eval_script.py --dataset CS --attack_type edge --df_size 10000 --start_seed 0 --end_seed 5 --db_name edge_cf_0.5 --cf 0.5 --log_name edge_logs --cacdc --contra_2 --yaum --gnndelete --retrain --gif --utu --scrub --megu