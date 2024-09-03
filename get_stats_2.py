import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data_from_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)["results"]
    return data


def create_dataframe(data):
    # Flattening the data into a DataFrame
    df_avg = pd.DataFrame(data["average"]).T
    df_std = pd.DataFrame(data["standard_dev"]).T

    # Combine average and standard deviation
    df_combined = pd.concat(
        [df_avg, df_std], axis=1, keys=["Average", "Standard Deviation"]
    )
    df_combined.columns = [
        "avg_time_taken",
        "avg_forget",
        "avg_utility",
        "std_time_taken",
        "std_forget",
        "std_utility",
    ]
    
    return df_combined


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
    dir = "logs"

    file_paths = []
    # get all files in dir and subdirs
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".json"):
                file_paths.append(os.path.join(root, file))

    for file_path in file_paths:
        data = load_data_from_json(file_path)

        # Create DataFrame
        df_combined = create_dataframe(data)

        # Display the DataFrame
        print(df_combined)

        # Create and display subplots
        fname = file_path.replace(".json", ".png")
        create_subplots(df_combined, fname)
