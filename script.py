import os
methods=["gnndelete", "gnndelete_ni", "gradient_ascent", "utu", "gif"]
df_sizes=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
datasets=["Cora_p", "Pubmed_p", "Citeseer_p"]

for dataset in datasets:
    for method in methods:
        for df_size in df_sizes:
            os.system(f"python .\main.py --df_size {df_size} --dataset {dataset} --unlearning_model {method} --attack_type edge")