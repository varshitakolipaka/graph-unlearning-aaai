# python hp_tune.py --unlearning_model gnndelete --dataset Cora
# python hp_tune.py --unlearning_model utu --dataset Cora
# python hp_tune.py --unlearning_model gif --dataset Cora
# python hp_tune.py --unlearning_model gradient_ascent --dataset Cora
# python hp_tune.py --unlearning_model scrub --dataset Cora
# python hp_tune.py --unlearning_model contrastive --dataset Cora

python hp_tune.py --unlearning_model utu --request edge --dataset Cora --edge_attack_type specific --attack_type edge --df_size 2000
python hp_tune.py --unlearning_model gif --request edge --dataset Cora --edge_attack_type specific  --attack_type edge --df_size 2000
python hp_tune.py --unlearning_model gradient_ascent --request edge --dataset Cora --edge_attack_type specific  --attack_type edge --df_size 2000
python hp_tune.py --unlearning_model contrastive --request edge --dataset Cora --edge_attack_type specific  --attack_type edge --df_size 2000

# python hp_tune.py --unlearning_model gnndelete --dataset Citeseer_p
# python hp_tune.py --unlearning_model utu --dataset Citeseer_p
# python hp_tune.py --unlearning_model gif --dataset Citeseer_p
# python hp_tune.py --unlearning_model gradient_ascent --dataset Citeseer_p
# python hp_tune.py --unlearning_model scrub --dataset Citeseer_p

# python hp_tune.py --unlearning_model gnndelete --dataset Amazon
# python hp_tune.py --unlearning_model utu --dataset Amazon
# python hp_tune.py --unlearning_model gif --dataset Amazon
# python hp_tune.py --unlearning_model gradient_ascent --dataset Amazon
# python hp_tune.py --unlearning_model scrub --dataset Amazon

# python hp_tune.py --unlearning_model contrastive --dataset Citeseer_p
# python hp_tune.py --unlearning_model contrastive --dataset Amazon