# # LINE PLOTS!
# import os
# import csv
# import argparse
# import matplotlib.pyplot as plt
# from collections import defaultdict

# # Read CSV function that reads the data and skips the header row
# def read_csv(file_path, metric="forget"):
#     data = defaultdict(lambda: "N/A")
#     with open(file_path, 'r') as f:
#         reader = csv.reader(f)
#         next(reader)  # Skip the header line
#         for row in reader:
#             method = row[0].lower()
#             data[method] = row[3] if metric == "forget" else row[1]
#     return data

# # Method aliases
# method_aliases = {
#     "yaum": "ACDC",
#     "cacdc": "SAGE-ACDC",
#     "utu": "UtU",
#     "megu": "MEGU",
#     "gnndelete": "GNNDelete",
#     "contra_2": "SAGE",
#     "original": "Oracle",
#     "poisoned": "Poisoned",
#     "retrain": "Retrain"
# }

# # Methods to plot in grayscale
# grayscale_methods = ["gif", "megu", "utu", "yaum", "contra_2", "scrub"]
# grayscale_linestyles = ['--', '-.', ':', (0, (3, 5, 1, 5)), (0, (5, 10)), (0, (1, 10))]

# # Plotting function
# def plot_metric(datasets, metric):
#     methods = list(next(iter(datasets.values())).keys())
#     x_values = sorted(datasets.keys(), key=lambda x: float(x))

#     # Prepare the figure and axes
#     plt.figure(figsize=(8,6))

#     # Prepare data for plotting
#     for idx, method in enumerate(methods):
#         y_values = [float(datasets[x][method].split('±')[0].strip()) * 100 if datasets[x][method] != "N/A" else None for x in x_values]
#         alias = method_aliases.get(method, method)

#         # Determine the color, line width, and line style
#         if method in grayscale_methods:
#             color = 'gray'
#             linestyle = grayscale_linestyles[idx % len(grayscale_linestyles)]
#             linewidth = 1.5  # Lighter gray lines
#         else:
#             color = None  # Let matplotlib assign a color
#             linestyle = '-'
#             linewidth = 2.5  # Thicker lines for non-gray

#         plt.plot(x_values, y_values, label=alias, color=color, linestyle=linestyle, linewidth=linewidth)

#     plt.xlabel('Fraction Size')
#     plt.ylabel(f'{metric.capitalize()} Score')
#     plt.title(f'{metric.capitalize()} Scores')
#     plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# # Main function to read data and generate the plot
# def main():
#     parser = argparse.ArgumentParser(description="Generate a line plot from CSV files")
#     parser.add_argument("--dir", default=".", help="Directory containing CSV files")
#     parser.add_argument("--metric", choices=["forget", "utility"], default="forget", help="Metric to plot (default: forget)")
#     args = parser.parse_args()

#     datasets = {}
#     for file in os.listdir(args.dir):
#         if file.endswith(".csv"):
#             dataset_name = file[:-4]  # Get the filename without extension to use as x-axis label
#             datasets[dataset_name] = read_csv(os.path.join(args.dir, file), metric=args.metric)

#     plot_metric(datasets, metric=args.metric)

# if __name__ == "__main__":
#     main()

# # SCATTER + LINE

# import os
# import csv
# import argparse
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import defaultdict

# # Read CSV function that reads the data and skips the header row
# def read_csv(file_path, metric="forget"):
#     data = defaultdict(lambda: "N/A")
#     with open(file_path, 'r') as f:
#         reader = csv.reader(f)
#         next(reader)  # Skip the header line
#         for row in reader:
#             method = row[0].lower()
#             data[method] = row[3] if metric == "forget" else row[1]
#     return data

# # Method aliases
# method_aliases = {
#     "yaum": "ACDC",
#     "cacdc": "SAGE-ACDC",
#     "utu": "UtU",
#     "megu": "MEGU",
#     "gnndelete": "GNNDelete",
#     "contra_2": "SAGE",
#     "original": "Oracle",
#     "poisoned": "Poisoned",
#     "retrain": "Retrain"
# }

# # Methods to skip
# skip_methods = ["contra_2", "yaum"]  # Skip SAGE and ACDC

# # Plotting function
# def plot_metric(datasets, metric):
#     methods = list(next(iter(datasets.values())).keys())
#     x_labels = sorted(datasets.keys(), key=lambda x: float(x))
#     x_values = np.array(range(len(x_labels)))  # Set x values for the fraction labels

#     # Prepare the figure and axes
#     plt.figure(figsize=(4, 6))

#     # Loop through methods to plot each as scatter points or lines
#     for method in methods:
#         if method in skip_methods:
#             continue

#         y_values = [float(datasets[label][method].split('±')[0].strip()) * 100 if datasets[label][method] != "N/A" else None for label in x_labels]
#         alias = method_aliases.get(method, method)

#         if method == "poisoned":
#             # Draw thick red line for 'Poisoned'
#             level = y_values[0] if y_values[0] is not None else 0
#             plt.axhline(y=level, color='red', linestyle='--', linewidth=3, label=alias)
#         elif method == "original":
#             # Draw thick green line for 'Oracle'
#             level = y_values[0] if y_values[0] is not None else 0
#             plt.axhline(y=level, color='green', linestyle='-', linewidth=3, label=alias)
#         elif method == "retrain":
#             # Draw gray line for 'Retrain'
#             plt.plot(x_values, y_values, linestyle='-', marker='o', linewidth=2.5, color='gray', label=alias)
#         elif method == "cacdc":
#             # Plot SAGE-ACDC with scatter points connected by a black line
#             plt.plot(x_values, y_values, marker='o', linestyle='-', linewidth=2.5, color='black', alpha=0.7, label=alias)
#         else:
#             # Plot other methods as scatter points in black
#             plt.scatter(x_values, y_values, label=alias, alpha=0.7)

#     plt.xlabel('Fraction Size')
#     plt.ylabel(f'{metric.capitalize()} Score')
#     plt.title(f'{metric.capitalize()} Scores')
#     plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels)  # Place the x-tick labels closer
#     plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.grid(axis='y')
#     plt.tight_layout()
#     plt.show()

# # Main function to read data and generate the plot
# def main():
#     parser = argparse.ArgumentParser(description="Generate a scatter plot from CSV files")
#     parser.add_argument("--dir", default=".", help="Directory containing CSV files")
#     parser.add_argument("--metric", choices=["forget", "utility"], default="forget", help="Metric to plot (default: forget)")
#     args = parser.parse_args()

#     datasets = {}
#     for file in os.listdir(args.dir):
#         if file.endswith(".csv"):
#             dataset_name = file[:-4]  # Get the filename without extension to use as x-axis label
#             datasets[dataset_name] = read_csv(os.path.join(args.dir, file), metric=args.metric)

#     plot_metric(datasets, metric=args.metric)

# if __name__ == "__main__":
#     main()


import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Read CSV function that reads the data and skips the header row
def read_csv(file_path, metric="forget"):
    data = defaultdict(lambda: "N/A")
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header line
        for row in reader:
            method = row[0].lower()
            data[method] = row[3] if metric == "forget" else row[1]
    return data

# Method aliases
method_aliases = {
    "yaum": "ACDC",
    "cacdc": "SAGE-ACDC",
    "utu": "UtU",
    "megu": "MEGU",
    "gnndelete": "GNNDelete",
    "contra_2": "SAGE",
    "original": "Oracle",
    "poisoned": "Poisoned",
    "retrain": "Retrain"
}

# Methods to exclude from the entire plot
general_exclude_methods = ["contra_2", "yaum"]

# Methods to skip when calculating next best
exclude_for_next_best = ["cacdc", "poisoned", "original", "retrain"]

# Plotting function
def plot_metric_emphasized(datasets, metric):
    methods = list(next(iter(datasets.values())).keys())
    x_labels = sorted(datasets.keys(), key=lambda x: float(x))
    x_values = np.array(range(len(x_labels)))  # Set x values for the fraction labels

    # Calculate the next best method (excluding specific methods)
    avg_scores = {}
    for method in methods:
        if method not in general_exclude_methods and method not in exclude_for_next_best:
            y_values = [float(datasets[label][method].split('±')[0].strip()) * 100 if datasets[label][method] != "N/A" else None for label in x_labels]
            avg_scores[method] = np.nanmean([y for y in y_values if y is not None])

    # Determine the next best method based on average score
    next_best_method = max(avg_scores, key=avg_scores.get)

    # Prepare the figure and axes
    plt.figure(figsize=(4, 6))

    # Define different markers for non-highlighted methods
    markers = ['o', 's', '^', 'v', '<', '>', 'D', 'p']
    marker_idx = 0

    # Loop through methods to plot each as a line
    for method in methods:
        if method in general_exclude_methods:
            continue

        y_values = [float(datasets[label][method].split('±')[0].strip()) * 100 if datasets[label][method] != "N/A" else None for label in x_labels]
        alias = method_aliases.get(method, method)

        if method == "poisoned":
            # Draw thick red dashed line for 'Poisoned'
            plt.axhline(y=y_values[0] if y_values[0] is not None else 0, color='red', linestyle='--', linewidth=3.5, label=alias)
        elif method == "original":
            # Draw thick green solid line for 'Oracle'
            plt.axhline(y=y_values[0] if y_values[0] is not None else 0, color='green', linestyle='-', linewidth=3.5, label=alias)
        elif method == "retrain":
            # Draw gray line for 'Retrain'
            plt.plot(x_values, y_values, linestyle='-', marker='o', linewidth=2.5, color='gray', alpha=0.7, label=alias)
        elif method == "cacdc":
            # Highlight SAGE-ACDC with a bold line and distinct color
            plt.plot(x_values, y_values, marker='o', linestyle='-', linewidth=2.5, color='blue', alpha=1.0, label=alias)
        elif method == next_best_method:
            # Highlight the next best method with a distinct color (orange)
            plt.plot(x_values, y_values, marker='o', linestyle='-', linewidth=2.5, color='orange', alpha=0.9, label=alias)
        else:
            # Plot other methods with lighter lines and different markers
            plt.plot(x_values, y_values, marker=markers[marker_idx % len(markers)], linestyle='--', linewidth=1, color='gray', alpha=0.5, label=alias)
            marker_idx += 1

    # Set labels and title
    plt.xlabel('Fraction Size')
    plt.ylabel(f'{metric.capitalize()} Score')
    plt.title(f'{metric.capitalize()} Scores')
    plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels)  # Place the x-tick labels closer
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# Main function to read data and generate the plot
def main():
    parser = argparse.ArgumentParser(description="Generate a line plot from CSV files with emphasis on best method")
    parser.add_argument("--dir", default=".", help="Directory containing CSV files")
    parser.add_argument("--metric", choices=["forget", "utility"], default="forget", help="Metric to plot (default: forget)")
    args = parser.parse_args()

    datasets = {}
    for file in os.listdir(args.dir):
        if file.endswith(".csv"):
            dataset_name = file[:-4]  # Get the filename without extension to use as x-axis label
            datasets[dataset_name] = read_csv(os.path.join(args.dir, file), metric=args.metric)

    plot_metric_emphasized(datasets, metric=args.metric)

if __name__ == "__main__":
    main()