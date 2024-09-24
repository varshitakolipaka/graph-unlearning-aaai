import json
import numpy as np
import os

def process_json_file(file_path):
    # Load the data from the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Function to calculate the averages and standard deviations
    def calculate_stats(results):
        # Automatically detect methods and keys
        methods = set()
        keys = set()
        
        for seed in results:
            if seed == "average" or seed == "standard_dev":
                continue
            for method in results[seed]:
                methods.add(method)
                for key in results[seed][method]:
                    keys.add(key)
        
        # Initialize containers for averages and standard deviations
        averages = {method: {key: [] for key in keys} for method in methods}
        std_devs = {method: {key: [] for key in keys} for method in methods}
        
        # Gather all values across seeds
        for seed in results:
            if seed == "average" or seed == "standard_dev":
                continue
            for method in methods:
                if method in results[seed]:
                    for key in results[seed][method]:
                        averages[method][key].append(results[seed][method][key])
        
        # Calculate averages and standard deviations
        for method in methods:
            for key in keys:
                if averages[method][key]:  # Ensure that there are values to average
                    std_devs[method][key] = np.std(averages[method][key])
                    averages[method][key] = np.mean(averages[method][key])
                else:  # Handle cases where no values are present
                    averages[method][key] = None
                    std_devs[method][key] = None
        
        return averages, std_devs

    # Get averages and standard deviations
    averages, std_devs = calculate_stats(data["results"])

    # Add averages and standard deviations to the original dictionary
    data["results"]["average"] = averages
    data["results"]["standard_dev"] = std_devs

    # Write the updated data back to the JSON file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# Example usage
if __name__ == "__main__":
    dir = 'logs'
    
    # get all json files in dir and subdirectories
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.json'):
                if "new_metric" in file:
                    continue
                print(f"Processing {file}")
                process_json_file(os.path.join(root, file))
