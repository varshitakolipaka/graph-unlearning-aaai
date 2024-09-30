import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_csv_files(folder, frac_sizes):
    """
    Loads all the relevant CSV files from the given folder.
    """
    csv_files = {}
    try:
        for file in os.listdir(folder):
            if ('run_logs_label_0.5_' in file or 'run_logs_edge' in file) and file.endswith('.csv'):
                frac_size = get_frac_size_from_filename(file)
                if frac_size in frac_sizes:
                    csv_files[frac_size] = pd.read_csv(os.path.join(folder, file))

        # Ensure all frac sizes have an entry, even if the file was not found
        for frac_size in frac_sizes:
            if frac_size not in csv_files:
                csv_files[frac_size] = None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return csv_files


def get_frac_size_from_filename(filename):
    """
    Extracts the fraction size from the filename.
    """
    if '_cf_' in filename:
        return float(filename.split('_cf_')[-1].replace('.csv', ''))
    else:
        return 1.0


def process_data(csv_files):
    """
    Processes the data from the loaded CSV files.
    """
    data = []
    all_methods = set()
    try:
        # First pass to collect all methods
        for df in csv_files.values():
            if df is not None:
                all_methods.update(df['Method'].unique())

        # Second pass to collect data, setting None for missing frac_sizes or methods
        for frac_size, df in csv_files.items():
            for method in all_methods:
                if df is None or method not in df['Method'].values:
                    data.append({'frac_size': frac_size, 'method': method, 'forget_mean': None, 'forget_std': None})
                else:
                    row = df[df['Method'] == method].iloc[0]
                    forget_mean, forget_std = parse_value_with_std(row['forget'])
                    data.append({'frac_size': frac_size, 'method': method, 'forget_mean': forget_mean, 'forget_std': forget_std})
    except KeyError as e:
        print(f"Error: Column {e} not found in the CSV file.")
    except Exception as e:
        print(f"An unexpected error occurred while processing the data: {e}")
    return pd.DataFrame(data)


def parse_value_with_std(value):
    """
    Parses a value with standard deviation formatted as 'mean ± std'.
    """
    try:
        mean, std = value.split(' ± ')
        return float(mean), float(std)
    except ValueError:
        return float(value), 0.0


def plot_forget_scores(df, folder):
    """
    Creates the line plot for the forget scores with error bars.
    """
    
    def assign_method_color(methods):
        """
        Assigns a color to the method.
        """
        colors = sns.color_palette('husl', n_colors=len(methods))
        
        # sort methods to ensure consistent color assignment
        methods = sorted(methods)
        
        return {method: colors[i] for i, method in enumerate(methods)}
    
    try:
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df,
            x='frac_size',
            y='forget_mean',
            hue='method',
            palette=assign_method_color(df['method'].unique()),
            style='method',
            markers=True,
            dashes=False
        )
        
        # add error bars
        # for method in df['method'].unique():
        #     method_data = df[df['method'] == method]
        #     plt.errorbar(
        #         method_data['frac_size'],
        #         method_data['forget_mean'],
        #         yerr=method_data['forget_std'],
        #         fmt='none',
        #         c=assign_method_color(df['method'].unique())[method],
        #         capsize=2
        #     )
        

        plt.xlabel('Fraction Size')
        plt.ylabel('Forget Score')
        plt.title(f'Forget Scores: {folder.split("/")[-1]}')
        plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(folder, 'forget_scores.png'))
    except KeyError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred while plotting: {e}")


def main(folder):
    """
    Main function to load, process, and plot data.
    """
    frac_sizes = [0.0, 0.05, 0.25, 0.5, 0.75, 1.0]
    csv_files = load_csv_files(folder, frac_sizes)

    if not csv_files:
        print("No valid CSV files found. Please check the directory.")
        return

    df = process_data(csv_files)
    if df.empty:
        print("No valid data to plot.")
        return

    plot_forget_scores(df, folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot forget scores across different fraction sizes.')
    parser.add_argument('--i', type=str, help='Folder containing the CSV files.')
    args = parser.parse_args()
    main(args.i)
