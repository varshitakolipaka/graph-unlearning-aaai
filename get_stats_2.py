import json
import csv
import os

def write_to_csv(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Convert to CSV format
    output_data = []
    for method, avg_metrics in data['results']['average'].items():
        std_metrics = data['results']['standard_dev'][method]
        output_data.append({
            'method': method,
            'avg. forget': avg_metrics['forget'],
            'std. forget': std_metrics['forget'],
            'avg. utility': avg_metrics['utility'],
            'std. utility': std_metrics['utility'],
            'avg. time': avg_metrics['time_taken'],
            'std. time': std_metrics['time_taken']
        })

    # Specify CSV file name
    csv_filename = file_path.replace('.json', '.csv')

    # Writing to CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['method', 'avg. forget', 'std. forget', 'avg. utility', 'std. utility', 'avg. time', 'std. time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in output_data:
            writer.writerow(row)

    print(f"Data has been written to {csv_filename}")
    

dir = 'logs'
# get all files in the directory and subdirectories
for root, dirs, files in os.walk(dir):
    for file in files:
        if file.endswith('.json'):
            file_path = os.path.join(root, file)
            write_to_csv(file_path)