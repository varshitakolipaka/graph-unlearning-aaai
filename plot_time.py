import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

method_to_name = {
    'cacdc': 'Cognac',
    'yaum': 'AC/DC',
    'gif': 'GIF',
    'gnndelete': 'GNNDelete',
    'retrain': 'Retrain',
    'scrub': 'SCRUB',
    'utu': 'UtU',
    'megu': 'MEGU'
}

# Function to plot the time taken by each method from a CSV file
def plot_time_taken(csv_file):
    # Read CSV file into DataFrame
    df = pd.read_csv(csv_file)
    
    sns.set_style('whitegrid')

    # Extract and convert time_taken to numeric values, ignoring uncertainty
    df['time_taken'] = pd.to_numeric(df['time_taken'].str.split('±').str[0], errors='coerce')

    # Filter out rows with missing or NaN time_taken values
    df = df.dropna(subset=['time_taken'])
    
    # change method names to human readable names, and if not found, delete the row
    df['Method'] = df['Method'].apply(lambda x: method_to_name.get(x, np.nan))

    # Plotting with numbers on top of the bars
    plt.figure(figsize=(6, 6))
    ax = sns.barplot(x='Method', y='time_taken', data=df, palette='viridis')
    # plt.xticks(rotation=45, ha='right')
    # plt.xlabel('Method')
    # plt.ylabel('Time Taken')
    plt.title('Time Taken')
    
    # increase size of x-axis labels
    plt.xticks(fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    
    plt.ylabel('Time Taken (s)', fontsize=12)
    plt.xlabel('')
    
    # Add numbers on top of the bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.3f}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('time_taken.png')

# Example usage
plot_time_taken('logs/label_cf_logs/Amazon/run_logs_label_0.5_3_4_cf_0.25.csv')
