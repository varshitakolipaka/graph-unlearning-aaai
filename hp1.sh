# python hp_tune.py --unlearning_model utu --dataset Cora --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data
# python hp_tune.py --unlearning_model utu --dataset Cora --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data
# python hp_tune.py --unlearning_model utu --dataset Cora --df_size 0.5 --random_seed 2 --data_dir /scratch/akshit.sinha/data
# python hp_tune.py --unlearning_model utu --dataset Cora --df_size 0.5 --random_seed 3 --data_dir /scratch/akshit.sinha/data
# python hp_tune.py --unlearning_model utu --dataset Cora --df_size 0.5 --random_seed 4 --data_dir /scratch/akshit.sinha/data

# python hp_tune.py --unlearning_model retrain --dataset Amazon --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data 
# python hp_tune.py --unlearning_model retrain --dataset Amazon --df_size 0.5 --random_seed 2 --data_dir /scratch/akshit.sinha/data
# python hp_tune.py --unlearning_model retrain --dataset Amazon --df_size 0.5 --random_seed 3 --data_dir /scratch/akshit.sinha/data
# python hp_tune.py --unlearning_model retrain --dataset Amazon --df_size 0.5 --random_seed 4 --data_dir /scratch/akshit.sinha/data
 
# python hp_tune.py --unlearning_model retrain --dataset PubMed --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data
# python hp_tune.py --unlearning_model retrain --dataset PubMed --df_size 0.5 --random_seed 2 --data_dir /scratch/akshit.sinha/data
# python hp_tune.py --unlearning_model retrain --dataset PubMed --df_size 0.5 --random_seed 3 --data_dir /scratch/akshit.sinha/data
# python hp_tune.py --unlearning_model retrain --dataset PubMed --df_size 0.5 --random_seed 4 --data_dir /scratch/akshit.sinha/data


python hp_tune.py --unlearning_model retrain --dataset PubMed --attack_type label  --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data
python hp_tune.py --unlearning_model retrain --dataset PubMed --attack_type label  --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data
python hp_tune.py --unlearning_model retrain --dataset PubMed --attack_type label  --df_size 0.5 --random_seed 2 --data_dir /scratch/akshit.sinha/data
python hp_tune.py --unlearning_model retrain --dataset PubMed --attack_type label  --df_size 0.5 --random_seed 3 --data_dir /scratch/akshit.sinha/data
python hp_tune.py --unlearning_model retrain --dataset PubMed --attack_type label  --df_size 0.5 --random_seed 4 --data_dir /scratch/akshit.sinha/data

python hp_tune.py --unlearning_model retrain --dataset Amazon --attack_type label  --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data
python hp_tune.py --unlearning_model retrain --dataset Amazon --attack_type label  --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data
python hp_tune.py --unlearning_model retrain --dataset Amazon --attack_type label  --df_size 0.5 --random_seed 2 --data_dir /scratch/akshit.sinha/data
python hp_tune.py --unlearning_model retrain --dataset Amazon --attack_type label  --df_size 0.5 --random_seed 3 --data_dir /scratch/akshit.sinha/data
python hp_tune.py --unlearning_model retrain --dataset Amazon --attack_type label  --df_size 0.5 --random_seed 4 --data_dir /scratch/akshit.sinha/data

python hp_tune.py --unlearning_model retrain --dataset Cora --attack_type label  --df_size 0.5 --random_seed 0 --data_dir /scratch/akshit.sinha/data
python hp_tune.py --unlearning_model retrain --dataset Cora --attack_type label  --df_size 0.5 --random_seed 1 --data_dir /scratch/akshit.sinha/data
python hp_tune.py --unlearning_model retrain --dataset Cora --attack_type label  --df_size 0.5 --random_seed 2 --data_dir /scratch/akshit.sinha/data
python hp_tune.py --unlearning_model retrain --dataset Cora --attack_type label  --df_size 0.5 --random_seed 3 --data_dir /scratch/akshit.sinha/data
python hp_tune.py --unlearning_model retrain --dataset Cora --attack_type label  --df_size 0.5 --random_seed 4 --data_dir /scratch/akshit.sinha/data
# python hp_tune.py --unlearning_model utu --dataset Cora --attack_type edge --request edge --df_size 5000 --random_seed 0
# python hp_tune.py --unlearning_model utu --dataset Amazon --attack_type edge --request edge --df_size 30000 --random_seed 0
# python hp_tune.py --unlearning_model utu --dataset PubMed --attack_type edge --request edge --df_size 25000 --random_seed 0