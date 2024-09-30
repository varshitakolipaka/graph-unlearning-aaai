import json
import pandas as pd
import os
import argparse

def create_summary_table(json_data):
    average_data = json_data.get('average', {})
    std_data = json_data.get('standard_dev', {})

    # Create a list of dictionaries to hold processed row data
    rows = []

    # Iterate through each method in average data
    for method, avg_values in average_data.items():
        row = {'Method': method}
        
        # Add each metric in 'average' ± 'standard_dev' format
        for key, avg_value in avg_values.items():
            if key == "utility_f1" or key == "forget_f1":
                continue
            
            if key not in std_data.get(method, {}):
                continue
            
            std_value = std_data[method].get(key, 0.0)
            # Handling null values
            if avg_value is None or std_value is None:
                value_str = 'N/A'
            else:
                value_str = f"{avg_value:.3f} ± {std_value:.3f}"
            
            row[key] = value_str

        rows.append(row)

    # Create a DataFrame from the rows and return it
    df = pd.DataFrame(rows)
    
    # sort df by method name 
    df = df.sort_values(by='Method')
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process JSON files to calculate averages and standard deviations')
    parser.add_argument('--dir', type=str, default='logs/trigger_logs', help='Directory containing JSON files')
    
    args = parser.parse_args()
    dir = args.dir


    # get all files in dir and subdirs

    files = []
    for root, _, filenames in os.walk(dir):
        for filename in filenames:
            files.append(os.path.join(root, filename))


    for file in files:
        if not file.endswith('.json'):
            continue
        # Load JSON data from a file
        with open(file, 'r') as f:
            json_data = json.load(f)

        # Create summary table
        try:
            summary_table = create_summary_table(json_data["results"])
        except KeyError:
            print(f"Skipping {file} as it does not contain 'results' key")
            continue

        # Print the summary table
        print(summary_table)

        # Optionally save the table to a CSV file
        file_name = file.replace('.json', '.csv')
        print(f"Saving summary table to {file_name}")
        summary_table.to_csv(file_name, index=False)