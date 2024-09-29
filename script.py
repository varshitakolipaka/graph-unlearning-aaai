import os

df_sizes=[0.1, 0.25]
# datasets=["Amazon", "Cora", "CS"]
# class_pairs=[(1, 6), (2, 5), (0, 7)]
# class_pairs= {"Amazon":[(1, 6), (2, 5), (0, 7)],
#               "Cora":[(50, 47), (69, 68), (19, 15)],
#               "CS":[(1,3), (5, 11), (7, 14)]}
datasets=["Cora"]
class_pairs= {"Cora":[(57, 68)]}
triggers=[15, 100]

for df_size in df_sizes:
    for dataset in datasets:
        for victim, target in class_pairs[dataset]:
            for trigger in triggers:
                for seed in range(5):
                    command= f"python ./tempmain.py --dataset {dataset} --df_size {df_size} --random_seed {seed} --victim_class {victim} --target_class {target} --trigger_size {trigger} "
                    os.system(command)