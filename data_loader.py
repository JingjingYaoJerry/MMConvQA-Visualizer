import json

def load_data(file_path='./data/MMCoQA_dev.txt'):
    """
    Load data from the MMCoQA JSON file.

    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        dict: Parsed JSON data.
    """
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file.readlines()]
    return data


if __name__ == "__main__":
    # Load the data
    data = load_data()
    # Print the second question in the first dialogue for verification
    print(data[1]['history'][0]['question'])