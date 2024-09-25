import os

df_sizes=[0.25, 0.5, 1]
class_pairs= [(50, 47), (69, 68), (19, 15)]
triggers=[15, 50, 100]

for seed in range(11):
    for df_size in df_sizes:
        for victim, target in class_pairs:
            for trigger in triggers:
                command= f"python ./tempmain.py --df_size {df_size} --random_seed {seed} --victim_class {victim} --target_class {target} --trigger_size {trigger} "
                os.system(command)