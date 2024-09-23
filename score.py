import json
import argparse
import numpy as np
import os

def normalize_score(method_score, poisoned_score, original_score):
    denominator = original_score - poisoned_score
    if denominator == 0:
        return np.nan  # Avoid division by zero
    print("numerator: ", (method_score - poisoned_score))
    print("denominator: ", (original_score - poisoned_score))
    print("score: ", ((method_score - poisoned_score) / denominator) * 100)
    return ((method_score - poisoned_score) / denominator) * 100

def process_json_data(data):
    results = data.get("results", {})
    normalized_scores = {}

    # Iterate through each seed in the results
    for seed_key, seed_data in results.items():
        # Skip non-seed entries like 'average' and 'standard_dev'
        if not seed_key.isdigit():
            continue

        # Extract oracle and poisoned scores
        try:
            original_forget = seed_data['original']['forget']
            poisoned_forget = seed_data['poisoned']['forget']
            original_utility = seed_data['original']['utility']
            poisoned_utility = seed_data['poisoned']['utility']
        except KeyError as e:
            print(f"Missing key in seed {seed_key}: {e}")
            continue

        # Iterate through each method in the seed
        for method, metrics in seed_data.items():
            print("SEED: ", seed_key)
            if method in ['original', 'poisoned']:
                continue  # Skip baseline methods

            # Ensure the method has 'forget' and 'utility' metrics
            if 'forget' not in metrics or 'utility' not in metrics:
                print(f"Missing metrics for method '{method}' in seed {seed_key}. Skipping.")
                continue

            print("Method: ", method)
            print("FORGET: (p, o, m)", poisoned_forget, original_forget, metrics['forget'])
            print("utility: (p, o, m)", poisoned_utility, original_utility, metrics['utility'])
            # Normalize 'forget' and 'utility' scores
            norm_forget = normalize_score(metrics['forget'], poisoned_forget, original_forget)
            norm_utility = normalize_score(metrics['utility'], poisoned_utility, original_utility)

            # Initialize method entry if not present
            if method not in normalized_scores:
                normalized_scores[method] = {'forget': [], 'utility': []}

            # Append normalized scores
            normalized_scores[method]['forget'].append(norm_forget)
            normalized_scores[method]['utility'].append(norm_utility)

    return normalized_scores

def aggregate_scores(normalized_scores):
    """
    Aggregate normalized scores to compute average and standard deviation.

    Args:
        normalized_scores (dict): Normalized scores for each method.

    Returns:
        dict: Aggregated average and standard deviation for each method.
    """
    aggregation = {}
    for method, metrics in normalized_scores.items():
        forget_scores = np.array(metrics['forget'])
        utility_scores = np.array(metrics['utility'])

        # Compute mean and std, ignoring NaN values
        forget_avg = np.nanmean(forget_scores)
        forget_std = np.nanstd(forget_scores)
        utility_avg = np.nanmean(utility_scores)
        utility_std = np.nanstd(utility_scores)

        aggregation[method] = {
            'forget_avg': forget_avg,
            'forget_std': forget_std,
            'utility_avg': utility_avg,
            'utility_std': utility_std
        }

    return aggregation

def main():
    parser = argparse.ArgumentParser(description="Normalize and aggregate scores from a single JSON file.")
    parser.add_argument(
        '--input_file',
        type=str,
        default='./logs_test/test_score.json',
        help='Path to the input JSON file containing results.'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Path to save the aggregated normalized results as a JSON file. If not provided, results are printed.'
    )
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    # Check if input file exists
    if not os.path.isfile(input_file):
        print(f"Input file '{input_file}' does not exist.")
        return

    # Load the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Process and normalize the scores
    normalized_scores = process_json_data(data)

    if not normalized_scores:
        print("No valid normalized scores found.")
        return

    # Aggregate the scores to compute average and standard deviation
    aggregated_results = aggregate_scores(normalized_scores)

    # Format the aggregated results for better readability
    formatted_results = {}
    for method, stats in aggregated_results.items():
        formatted_results[method] = {
            'forget_avg': round(stats['forget_avg'], 2),
            'forget_std': round(stats['forget_std'], 2),
            'utility_avg': round(stats['utility_avg'], 2),
            'utility_std': round(stats['utility_std'], 2)
        }

    # Output the results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(formatted_results, f, indent=4)
        print(f"Aggregated normalized results saved to '{output_file}'.")
    else:
        print("\nAggregated Normalized Scores:")
        for method, stats in formatted_results.items():
            print(f"Method: {method}")
            print(f"  Forget: {stats['forget_avg']} ± {stats['forget_std']}")
            print(f"  Utility: {stats['utility_avg']} ± {stats['utility_std']}\n")

if __name__ == "__main__":
    main()
