# python run_hp_tune.py --dataset Amazon --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label_strong --db_name strong_label --gnn gcn --cacdc
# python eval_script.py --dataset Amazon --df_size 0.5 --attack_type label_strong --cacdc --yaum
python run_hp_tune.py --dataset Cora --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label_strong --db_name strong_label --gnn gcn --cacdc --yaum
python eval_script.py --dataset Cora --df_size 0.5 --attack_type label_strong --cacdc --yaum
python run_hp_tune.py --dataset CS --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data --attack_type label_strong --db_name strong_label --gnn gcn --cacdc --yaum
python eval_script.py --dataset CS --df_size 0.5 --attack_type label_strong --cacdc --yaum


# python main.py --attack_type label_strong --dataset Amazon --random_seed 1 --data_dir /scratch/akshit.sinha/data --unlearning_model yaum
# python main.py --attack_type label_strong --dataset Amazon --random_seed 1 --data_dir /scratch/akshit.sinha/data --unlearning_model cacdc
# python main.py --attack_type label_strong --dataset Cora --random_seed 2 --data_dir /scratch/akshit.sinha/data --unlearning_model yaum
# python main.py --attack_type label_strong --dataset Cora --random_seed 2 --data_dir /scratch/akshit.sinha/data --unlearning_model cacdc
# python main.py --attack_type label_strong --dataset Cora --random_seed 3 --data_dir /scratch/akshit.sinha/data --unlearning_model yaum
# python main.py --attack_type label_strong --dataset Cora --random_seed 3 --data_dir /scratch/akshit.sinha/data --unlearning_model cacdc
# python main.py --attack_type label_strong --dataset Cora --random_seed 4 --data_dir /scratch/akshit.sinha/data --unlearning_model yaum
# python main.py --attack_type label_strong --dataset Cora --random_seed 4 --data_dir /scratch/akshit.sinha/data --unlearning_model cacdc
# python main.py --attack_type label_strong --dataset Cora --random_seed 5 --data_dir /scratch/akshit.sinha/data --unlearning_model yaum
# python main.py --attack_type label_strong --dataset Cora --random_seed 5 --data_dir /scratch/akshit.sinha/data --unlearning_model cacdc