import os

dir = '/scratch/akshit.sinha/data'

# get all files in the directory
files = os.listdir(dir)

for f in files:
    
    if 'model' not in f:
        continue
    
    split = f.split('_')
    
    split = ['gcn'] + split
    print(split)
    
    # rename the file
    os.rename(f'{dir}/{f}', f'{dir}/{"_".join(split)}')
