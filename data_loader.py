import json
import pandas as pd


def load_data(file_path: str) -> list[dict]:
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

def group_by_conversation(q_data: list[dict]) -> dict[str, list[dict]]:
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

def construct_lookups(evidence_data: list[dict]) -> dict[str, dict]:
    """
    Construct a lookup dictionary for fast access to evidence content by 'id'.

    Args:
        evidence_data (list): List of JSON-format evidence data.

    Returns:
        Dictionary mapping evidences' IDs to their content.
    """
    return {item['id']: item for item in evidence_data}

def construct_table_from_lookups(tab_json: dict[str, dict]) -> pd.DataFrame:
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

def prepare_all_data(qs_path: str, imgs_path: str, tabs_path: str, txts_path: str) -> tuple[dict[str, list[dict]], dict[str, dict], dict[str, dict], dict[str, dict]]:
    """
    Prepares all data for images, tables, and texts against each question.

    Args:
        qs_path (str): Path to the file associating questions with their evidences' doc IDs.
        imgs_path (str): Path to the file associating images with their doc IDs.
        tabs_path (str): Path to the file associating tables with their doc IDs.
        txts_path (str): Path to the file associating texts with their doc IDs.

    Returns:
        tuple(dict, dict, dict, dict): Grouped conversations, image lookups, table lookups, and text lookups.
    """
    print("Step 1: Loading all raw data files...")
    qs_data = load_data(qs_path)
    imgs_data = load_data(imgs_path)
    tabs_data = load_data(tabs_path)
    txts_data = load_data(txts_path)

    print("Step 2: Grouping questions into conversations...")
    convs = group_by_conversation(q_data)

    print("Step 3: Constructing lookups for all modalities' evidences...")
    imgs_lookups = construct_lookups(imgs_data)
    tabs_lookups = construct_lookups(tabs_data)
    txts_lookups = construct_lookups(txts_data)

    print("Data Preparation Complete.\n")
    return convs, imgs_lookups, tabs_lookups, txts_lookups


if __name__ == "__main__":
    # Test the preparation of data with all modalities
    print("Preparing all data for MMCoQA...")
    qs_path = r'.\data\MMCoQA_dev.txt'
    imgs_path = r'.\data\multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl'
    tabs_path = r'.\data\multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl'
    txts_path = r'.\data\multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl'
    convs, imgs_lookups, tabs_lookups, txts_lookups = prepare_all_data(qs_path, imgs_path, tabs_path, txts_path)
    print(f"Loaded {len(convs)} conversations.")
    print(f"Indexed {len(imgs_lookups)} image metadata.")
    print(f"Indexed {len(tabs_lookups)} tables metadata.")
    print(f"Indexed {len(txts_lookups)} text passages metadata.")
    # Demo: Retrieve the image metadata from conversation C_381
    print("Demo: Retrieve the first image metadata from conversation C_381...")
    turn = convs['C_381'][0]
    img_id = turn['answer'][0]['image_instances'][0]['doc_id']
    print(f"\nExample image metadata retrieval for ID {img_id}:")
    print(imgs_lookups[img_id])