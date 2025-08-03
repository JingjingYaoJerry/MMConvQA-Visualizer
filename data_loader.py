import json
import pandas as pd


def load_data(file_path: str) -> list:
    """
    Generic loader for data stored in lines in JSON format.

    Args:
        file_path (str): Path to the file.

    Returns:
        data (list): Parsed JSON-format data.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file: # 'utf-8' for handling jsonl files
            data = [json.loads(line) for line in file.readlines()]
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return []

def group_by_conversation(q_data):
    """
    Groups the flat list of questions into conversations based on their 'qid' prefix.

    Args:
        data (list): List of JSON-format questions loaded from the MMCoQA .txt file.

    Returns:
        conversations (dict): Grouped conversations by their ID.
    """
    conversations = {}
    for q in q_data:
        # According to the qid, the first two sections appear to be the conversation ID
        conv_id = q['qid'].split('_')[0] + '_' + q['qid'].split('_')[1] # e.g., "C_381_1" -> "C_381"
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(q)        
    return conversations

def construct_lookups(evidence_data: list) -> dict:
    """
    Construct a lookup dictionary for fast access to evidence content by 'id'.

    Args:
        evidence_data (list): List of JSON-format evidence data.

    Returns:
        Dictionary mapping evidences' IDs to their content.
    """
    return {item['id']: item for item in evidence_data}

def construct_table_from_lookups(tab_json: dict) -> pd.DataFrame:
    """
    Construct a table for a JSON-format table evidence in a DataFrame format.

    Args:
        tab_dict (dict): Dictionary mapping table IDs to their content.

    Returns:
        df (pd.DataFrame): DataFrame containing the table data.
    """
    headers = [header['column_name'] for header in tab_json['header']]
    rows = [[cell['text'] for cell in row] for row in tab_json['table_rows']]
    df = pd.DataFrame(rows, columns=headers)
    return df

def prepare_all_data(q_path, img_path, tab_path, txt_path):
    """
    Prepares all data for images, tables, and texts against each question.

    Args:
        q_path (str): Path to the file associating questions with their evidences' doc IDs.
        img_path (str): Path to the file associating images with their doc IDs.
        tab_path (str): Path to the file associating tables with their doc IDs.
        txt_path (str): Path to the file associating texts with their doc IDs.

    Returns:
        prepared_data (dict, dict, dict): 
    """
    print("Step 1: Loading all raw data files...")
    q_data = load_data(q_path)
    img_data = load_data(img_path)
    tab_data = load_data(tab_path)
    txt_data = load_data(txt_path)

    print("Step 2: Grouping questions into conversations...")
    convs = group_by_conversation(q_data)

    print("Step 3: Constructing lookups for all modalities' evidences...")
    img_lookups = construct_lookups(img_data)
    tab_lookups = construct_lookups(tab_data)
    txt_lookups = construct_lookups(txt_data)
    
    print("Data Preparation Complete.\n")
    return convs, img_lookups, tab_lookups, txt_lookups


if __name__ == "__main__":
    q_path = r'.\data\MMCoQA_dev.txt'
    img_path = r'.\data\multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl'
    tab_path = r'.\data\multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl'
    txt_path = r'.\data\multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl'
    # Test the preparer
    convs, img_lookups, tab_lookups, txt_lookups = prepare_all_data(q_path, img_path, tab_path, txt_path)
    print(f"Loaded {len(convs)} conversations.")
    print(f"Indexed {len(img_lookups)} image metadata.")
    print(f"Indexed {len(tab_lookups)} tables metadata.")
    print(f"Indexed {len(txt_lookups)} text passages metadata.")
    # Demo: Retrieve the image metadata from conversation C_381
    print("Demo: Retrieve the first image metadata from conversation C_381...")
    turn = convs['C_381'][0]
    img_id = turn['answer'][0]['image_instances'][0]['doc_id']
    print(f"\nExample image metadata retrieval for ID {img_id}:")
    print(img_lookups[img_id])