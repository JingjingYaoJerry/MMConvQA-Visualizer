import json

def load_data(file_path='./data/data.json'):
    """
    Load data from the MMConvQA JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        dict: Parsed JSON data.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


if __name__ == "__main__":
    # Load the data
    data = load_data()
    # Print the first question in the first dialogue for verification
    print(data[0]['dialogue'][0]['question'])