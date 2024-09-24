def categorize_and_group_classes(classes, counts):
    min_count, max_count = counts.min().item(), counts.max().item()
    low_threshold = min_count + (max_count - min_count) / 3
    high_threshold = max_count - (max_count - min_count) / 3
    low_classes = []
    medium_classes = []
    high_classes = []

    for i, count in enumerate(counts):
        if count <= low_threshold:
            low_classes.append((classes[i], count))
        elif count >= high_threshold:
            high_classes.append((classes[i], count))
        else:
            medium_classes.append((classes[i], count))
    return low_classes, medium_classes, high_classes

def find_min_diff_in_group(class_group):
    if len(class_group) < 2:
        return None
    min_diff_pair = None
    min_diff = float('inf')
    for i in range(len(class_group)):
        for j in range(i + 1, len(class_group)):
            class1, count1 = class_group[i]
            class2, count2 = class_group[j]
            diff = abs(count1 - count2)
            if diff < min_diff:
                min_diff = diff
                min_diff_pair = (class1, class2, diff)
    return min_diff_pair

def classes_by_varying_size(classes, counts):
    '''
    Function uses classes of low, medium or high size
    Within each category, it returns classes having least difference
    '''
    low_classes, medium_classes, high_classes = categorize_and_group_classes(classes, counts)
    low_pair = find_min_diff_in_group(low_classes)
    medium_pair = find_min_diff_in_group(medium_classes)
    high_pair = find_min_diff_in_group(high_classes)
    return {'low': low_pair, 'medium': medium_pair, 'high': high_pair}


def classes_by_varying_difference(classes, counts):

    '''
    Returns the classes by varying the difference between classes
    Choose the following pairs:
    Low Difference: Largest and Second Largest classes
    Medium Difference: Largest and a Medium sized classes
    High Difference: Largest and Smallest classes
    '''

    class_count_pairs = sorted(zip(classes, counts), key=lambda x: x[1])
    smallest_class = class_count_pairs[0] if class_count_pairs else None
    largest_class = class_count_pairs[-1] if class_count_pairs else None
    second_largest_class = class_count_pairs[-2] if len(class_count_pairs) > 1 else None
    medium_class = class_count_pairs[len(class_count_pairs) // 2] if len(class_count_pairs) > 2 else None

    results = {}
    if largest_class and smallest_class:
        results['high'] = (largest_class[0], smallest_class[0])
    if largest_class and medium_class:
        results['medium'] = (largest_class[0], medium_class[0])
    if largest_class and second_largest_class:
        results['low'] = (largest_class[0], second_largest_class[0])

    return results if results else 'Not enough classes to form all required pairs.'