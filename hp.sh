# python hp_tune.py --unlearning_model scrub --dataset Cora --df_size 0.5 --random_seed 0
# python hp_tune.py --unlearning_model megu --dataset Cora --df_size 0.5 --random_seed 0
# python hp_tune.py --unlearning_model gnndelete --dataset Cora --df_size 0.5 --random_seed 0
# python hp_tune.py --unlearning_model utu --dataset Cora --df_size 0.5 --random_seed 0
# python hp_tune.py --unlearning_model gif --dataset Cora --df_size 0.5 --random_seed 0
# python hp_tune.py --unlearning_model gradient_ascent --dataset Cora --df_size 0.5 --random_seed 0
# python hp_tune.py --unlearning_model contrastive --dataset Cora --df_size 0.5 --random_seed 0
# python hp_tune.py --unlearning_model retrain --dataset Cora --df_size 0.5 --random_seed 0

# python hp_tune.py --unlearning_model utu --request edge --dataset Cora --edge_attack_type specific --attack_type edge --df_size 2000
# python hp_tune.py --unlearning_model gif --request edge --dataset Cora --edge_attack_type specific  --attack_type edge --df_size 2000
# python hp_tune.py --unlearning_model gradient_ascent --request edge --dataset Cora --edge_attack_type specific  --attack_type edge --df_size 2000
# python hp_tune.py --unlearning_model contrastive --request edge --dataset Cora --edge_attack_type specific  --attack_type edge --df_size 2000

# python hp_tune.py --unlearning_model utu --dataset Cora --attack_type edge --request edge --df_size 5000 --random_seed 0
# python hp_tune.py --unlearning_model scrub --dataset Cora --attack_type edge --request edge --df_size 5000 --random_seed 0
# python hp_tune.py --unlearning_model gif --dataset Cora --attack_type edge --request edge --df_size 5000 --random_seed 0
# python hp_tune.py --unlearning_model gnndelete --dataset Cora --attack_type edge --request edge --df_size 5000 --random_seed 0
# python hp_tune.py --unlearning_model contrastive --dataset Cora --attack_type edge --request edge --df_size 5000 --random_seed 0
# python hp_tune.py --unlearning_model gradient_ascent --dataset Cora --attack_type edge --request edge --df_size 5000 --random_seed 0
# python hp_tune.py --unlearning_model megu --dataset Cora --attack_type edge --request edge --df_size 5000 --random_seed 0
# python hp_tune.py --unlearning_model retrain --dataset Cora --attack_type edge --request edge --df_size 5000 --random_seed 0

# python hp_tune.py --unlearning_model gnndelete --dataset PubMed --df_size 0.5 --random_seed 0
# python hp_tune.py --unlearning_model utu --dataset PubMed --df_size 0.5 --random_seed 0
# python hp_tune.py --unlearning_model gif --dataset PubMed --df_size 0.5 --random_seed 0
# python hp_tune.py --unlearning_model gradient_ascent --dataset PubMed --df_size 0.5 --random_seed 0
# python hp_tune.py --unlearning_model scrub --dataset PubMed --df_size 0.5 --random_seed 0
python hp_tune.py --unlearning_model contrastive --dataset PubMed --df_size 0.5 --random_seed 0

# python hp_tune.py --unlearning_model gnndelete --dataset Amazon
# python hp_tune.py --unlearning_model utu --dataset Amazon
# python hp_tune.py --unlearning_model gif --dataset Amazon
# python hp_tune.py --unlearning_model gradient_ascent --dataset Amazon
# python hp_tune.py --unlearning_model scrub --dataset Amazon

# python hp_tune.py --unlearning_model contrastive --dataset Amazon