import json

def load_data(file_path='./data/MMCoQA_dev.txt'):
    """
    Load data from the MMCoQA .txt file where each line is in JSON format.

    Args:
        file_path (str): Path to the .txt file.

    Returns:
        list: Parsed JSON data.
    """
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file.readlines()]
    return data


if __name__ == "__main__":
    # Test the loader
    data = load_data()
    print(f"Loaded {len(data)} questions.")
    # Print the second question in the first dialogue for verification
    print(data[1]['history'][0]['question'])