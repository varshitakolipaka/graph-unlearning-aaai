import json
import numpy as np
import os

def forget_metric(method_acc, poison_acc, oracle_acc):
    print("method_acc", method_acc)
    print("poison_acc", poison_acc)
    print("oracle_acc", oracle_acc)
    
    print("method_acc - poison_acc", method_acc - poison_acc)
    print("oracle_acc - poison_acc", oracle_acc - poison_acc)
    
    print("(method_acc - poison_acc) / (oracle_acc - poison_acc)", (method_acc - poison_acc) / (oracle_acc - poison_acc))
    
    return 100 * (method_acc - poison_acc) / (oracle_acc - poison_acc) 

def util_metric_oracle(method_acc, oracle_acc):
    return 100 * (method_acc - oracle_acc) / oracle_acc

def util_metric_poisoned(method_acc, poisoned_acc):
    return 100 * (method_acc - poisoned_acc) / poisoned_acc

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
                    
        
        keys.add("util_oracle")
        keys.add("util_poisoned")
        
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
                        if key == "forget":
                            averages[method][key].append(forget_metric(results[seed][method][key], results[seed]["poisoned"][key], results[seed]["original"][key]))
                        elif key == "utility":
                            averages[method]["util_oracle"].append(util_metric_oracle(results[seed][method][key], results[seed]["original"][key]))
                            averages[method]["util_poisoned"].append(util_metric_poisoned(results[seed][method][key], results[seed]["poisoned"][key]))
                        else:
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
    file_path = file_path.split(".json")[0] + "_new_metric.json"
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# Example usage
if __name__ == "__main__":
    dir = 'logs'
    
    # get all json files in dir and subdirectories
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.json'):
                print(f"Processing {file}")
                try:
                    process_json_file(os.path.join(root, file))
                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    continue
