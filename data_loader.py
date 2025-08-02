import json


def load_data(file_path=r'.\data\MMCoQA_dev.txt'):
    """
    Load data from the MMCoQA .txt file where each line is in JSON format.

    Args:
        file_path (str): Path to the .txt file.

    Returns:
        data (list): Parsed JSON-format questions.
    """
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file.readlines()]
    return data

def group_data_by_conversation(data):
    """
    Groups the flat list of questions into conversations based on their qid prefix.

    Args:
        data (list): List of JSON-format questions loaded from the MMCoQA .txt file.

    Returns:
        conversations (dict): Grouped conversations by their ID.
    """
    conversations = {}
    for q in data:
        # According to the qid, the first two sections appear to be the conversation ID
        conv_id = q['qid'].split('_')[0] + '_' + q['qid'].split('_')[1] # e.g., "C_381_1" -> "C_381"
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(q)        
    return conversations


if __name__ == "__main__":
    # Test the loader
    data = load_data(file_path=r'./data/MMCoQA_dev.txt')
    print(f"Loaded {len(data)} questions.")
    grouped_conversations = group_data_by_conversation(data)
    print(f"Found {len(grouped_conversations)} unique conversations.")
    # Print turns in conversation C_381 for verification
    print(f"Conversation C_381 has {len(grouped_conversations['C_381'])} turns as follows:")
    for turn in grouped_conversations['C_381']:
        print(turn)
        print()