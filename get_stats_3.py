import pandas as pd

dir = "logs/Amazon"
attack = "edge"
# Load avg and std csv files
avg_df = pd.read_csv(f'{dir}/run_logs_{attack}_avg.csv')
std_df = pd.read_csv(f'{dir}/run_logs_{attack}_std.csv')

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
print(f'=========RESULTS:{dir.split("/")[-1]}-{attack}=========')
print(combined_df[['Method'] + metrics])

# Save to a new CSV if needed
combined_df.to_csv(f'{dir}/combined_table.csv', index=False)
