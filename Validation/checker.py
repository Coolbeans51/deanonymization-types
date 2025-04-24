def read_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        return [line.strip() for line in lines if line.strip()]

def calculate_accuracy(file1_data, file2_data):
    total = min(len(file1_data), len(file2_data))
    matches = 0

    for line1, line2 in zip(file1_data, file2_data):
        if line1 == line2:
            matches += 1

    accuracy = (matches / total) * 100 if total > 0 else 0
    return accuracy, matches, total

def main():
    file1_path = 'validation_seed_mapping.txt'
    file2_path = 'node_mapping.txt'

    data1 = read_file(file1_path)
    data2 = read_file(file2_path)

    accuracy, matches, total = calculate_accuracy(data1, data2)

    print(f"Matching lines: {matches}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
