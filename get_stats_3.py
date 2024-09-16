import os
import pandas as pd

dir = 'logs'

# get all subdirectories
subdirs = [f.path for f in os.scandir(dir) if f.is_dir()]

print(subdirs)

attacks = ['label', 'edge']

for subdir in subdirs:
    path = f'{subdir}'
    for attack in attacks:
        # Load avg and std csv files
        avg_df = pd.read_csv(f'{path}/run_logs_{attack}_avg.csv')
        std_df = pd.read_csv(f'{path}/run_logs_{attack}_std.csv')

        # Ensure both files have the same methods
        assert avg_df['Method'].equals(std_df['Method']), "Methods in avg and std CSV files do not match."

        # Combine avg and std into one DataFrame with format "value ± std"
        combined_df = avg_df.copy()
        metrics = ['Forget', 'Time Taken', 'Utility']

        # Format the table as "avg ± std"
        for metric in metrics:
            combined_df[metric] = avg_df[metric].round(3).astype(str) + ' ± ' + std_df[metric].round(3).astype(str)
            
        # sort methods by alphabetical order
        combined_df = combined_df.sort_values(by='Method')

        # Display the combined table
        print(f'=========RESULTS:{path.split("/")[-1]}-{attack}=========')
        print(combined_df[['Method'] + metrics])

        # Save to a new CSV if needed
        combined_df.to_csv(f'{path}/combined_table_{attack}.csv', index=False)
