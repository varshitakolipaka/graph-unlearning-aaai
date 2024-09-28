import csv
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data_from_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)["results"]
    return data

def convert_dict_to_csv(data, csv_file):
    """
    Convert a dictionary to a CSV file.

    Parameters:
    data (dict): The dictionary containing the data.
    csv_file (str): The name of the output CSV file.
    """
    with open(f"{csv_file}_avg.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(["Method", "Forget", "Time Taken", "Utility"])
        
        # Write the data
        for method, metrics in data["average"].items():
            writer.writerow([method, metrics["forget"], metrics["time_taken"], metrics["utility"]])
            
    with open(f"{csv_file}_std.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(["Method", "Forget", "Time Taken", "Utility"])
        
        # Write the data
        for method, metrics in data["standard_dev"].items():
            writer.writerow([method, metrics["forget"], metrics["time_taken"], metrics["utility"]])

    print(f"Data has been written to {csv_file}")

def create_subplots(df_combined, fname):
    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    
    # Utility plot
    sns.barplot(x=df_combined.index, y="avg_utility", data=df_combined, ax=axs[0])
    axs[0].errorbar(df_combined.index, df_combined["avg_utility"], yerr=df_combined["std_utility"], fmt='none', c='black', capsize=5)
    axs[0].set_title("Utility with Standard Deviation")
    axs[0].set_ylabel("Utility")
    axs[0].tick_params(axis='x', rotation=45)
    
    # Time taken plot
    sns.barplot(x=df_combined.index, y="avg_time_taken", data=df_combined, ax=axs[1])
    axs[1].errorbar(df_combined.index, df_combined["avg_time_taken"], yerr=df_combined["std_time_taken"], fmt='none', c='black', capsize=5)
    axs[1].set_title("Time Taken with Standard Deviation")
    axs[1].set_ylabel("Time Taken (seconds)")
    axs[1].tick_params(axis='x', rotation=45)
    
    # Forget plot
    sns.barplot(x=df_combined.index, y="avg_forget", data=df_combined, ax=axs[2])
    axs[2].errorbar(df_combined.index, df_combined["avg_forget"], yerr=df_combined["std_forget"], fmt='none', c='black', capsize=5)
    axs[2].set_title("Forget Ability with Standard Deviation")
    axs[2].set_ylabel("Forget")
    axs[2].tick_params(axis='x', rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(fname)


if __name__ == "__main__":
    # Load data from JSON
    dir = "logs/corrective"

    file_paths = []
    # get all files in dir and subdirs
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".json"):
                file_paths.append(os.path.join(root, file))

    for file_path in file_paths:
        data = load_data_from_json(file_path)
        convert_dict_to_csv(data, file_path.replace(".json", ""))