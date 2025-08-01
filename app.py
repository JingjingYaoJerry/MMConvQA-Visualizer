import streamlit as st
import pandas as pd
from os import path
from PIL import Image

from data_loader import load_data
from clip_analyzer import get_similarity

DATA_PATH = '.\data\MMCoQA_dev.txt'
IMG_DIR = r'.\data\final_dataset_images'

st.set_page_config(page_title='MMCoQA Explorer', layout='wide', initial_sidebar_state='auto')
st.title("MMCoQA Explorer")

# Load and cache the data to avoid reloading on every interaction
@st.cache_data
def cache_data(file_path):
    return load_data(file_path)

data = cache_data(DATA_PATH)
# Display for investigation
# data = {"qid": "C_802_8", "question": "What actor who made his feature film debut in the film \"Afterschool\" stared in Fantastic Beasts and Where to Find Them and stars Eddie Redmayne as Newt ?", "gold_question": "What actor who made his feature film debut in the film \"Afterschool\" stared in Fantastic Beasts and Where to Find Them and stars Eddie Redmayne as Newt ?", "answer": [{"answer": "Ezra Matthew Miller", "type": "string", "modality": "text", "text_instances": [{"doc_id": "5befee28f50f55b179e38b9a15c2a5b7", "part": "text", "start_byte": 0, "text": "Ezra Matthew Miller"}], "table_indices": [], "image_instances": []}], "question_type": "text", "table_id": "5690429e68c208d9327cbc06d1c46fc9", "history": [{"question": "In the first round of the 1976\u201377 Cypriot Cup, who did AEM Morphou play against?", "answer": [{"answer": "(A) EPA Larnaca", "type": "string", "modality": "table", "text_instances": [], "table_indices": [[11, 0]], "image_instances": []}]}, {"question": "\"Something's Going On\" is the third and final single by rock band, released from their album \"Hi-Fi Serious\", it is used in the What's New, Scooby-Doo?, an American animated sitcom mystery comedy series produced by who?", "answer": [{"answer": "Warner Bros.", "type": "string", "modality": "text", "text_instances": [{"doc_id": "d79f43a6298171fffebac03ce4b6cec4", "part": "text", "start_byte": 89, "text": "Warner Bros"}], "table_indices": [], "image_instances": []}]}, {"question": "Beasts of Balance is a dexterity tabletop game which is played alongside a companion app for iOS and Android, it was originally titled \"Fabulous Beasts\", the game had to be renamed following a trademark dispute with this entertainment company. over which 2016 fantasy film directed by David Yates?", "answer": [{"answer": "Fantastic Beasts and Where to Find Them", "type": "string", "modality": "text", "text_instances": [{"doc_id": "32afa0e94bdf18e724a223c8ee78c67f", "part": "text", "start_byte": 0, "text": "Fantastic Beasts and Where to Find Them"}], "table_indices": [], "image_instances": []}]}, {"question": "Chuck Jones directed a version of the entertainment company of Jack and the Beanstalk in 1955. Elmer Fudd was the giant, and who played Jack?", "answer": [{"answer": "Daffy Duck", "type": "string", "modality": "text", "text_instances": [{"doc_id": "e6daca9430b8cf9efc2e964b1fc9be70", "part": "text", "start_byte": 193, "text": "Daffy Duck"}, {"doc_id": "32d826b4b89b1815d40b99556b94859c", "part": "text", "start_byte": 349, "text": "Daffy Duck"}], "table_indices": [], "image_instances": []}]}, {"question": "Which Syncopy Inc. movie title(s) that has/have a tall building in the background of its poster?", "answer": [{"answer": "The Dark Knight (film)", "type": "string", "modality": "image", "text_instances": [], "table_indices": [], "image_instances": [{"doc_id": "f60afcdec9238132fc0f6d11e54c6457", "doc_part": "image"}]}, {"answer": "Inception", "type": "string", "modality": "image", "text_instances": [], "table_indices": [], "image_instances": [{"doc_id": "3f02d360ef0bf4ea2af7e0e7602fe5a7", "doc_part": "image"}]}]}, {"question": "Which Syncopy Inc. movie title(s) distributed by the entertainment company?", "answer": [{"answer": "Batman Begins", "type": "string", "modality": "table", "text_instances": [], "table_indices": [[0, 1]], "image_instances": []}, {"answer": "The Dark Knight", "type": "string", "modality": "table", "text_instances": [], "table_indices": [[2, 1]], "image_instances": []}, {"answer": "Inception", "type": "string", "modality": "table", "text_instances": [], "table_indices": [[3, 1]], "image_instances": []}, {"answer": "The Dark Knight Rises", "type": "string", "modality": "table", "text_instances": [], "table_indices": [[4, 1]], "image_instances": []}, {"answer": "Man of Steel", "type": "string", "modality": "table", "text_instances": [], "table_indices": [[5, 1]], "image_instances": []}, {"answer": "Dunkirk", "type": "string", "modality": "table", "text_instances": [], "table_indices": [[7, 1]], "image_instances": []}, {"answer": "Tenet", "type": "string", "modality": "table", "text_instances": [], "table_indices": [[8, 1]], "image_instances": []}]}, {"question": "Which Syncopy Inc. movie title(s) distributed by the entertainment company and that has/have a tall building in the background of its poster?", "answer": [{"answer": "The Dark Knight", "type": "string", "modality": "table", "text_instances": [], "table_indices": [[2, 1]], "image_instances": []}, {"answer": "Inception", "type": "string", "modality": "table", "text_instances": [], "table_indices": [[3, 1]], "image_instances": []}]}, {"question": "What film was directed by a British film editor and was a 2010 British-American fantasy film distributed by the entertainment company?", "answer": [{"answer": "Harry Potter and the Deathly Hallows \u2013 Part 1", "type": "string", "modality": "text", "text_instances": [{"doc_id": "e41c9c34f223e7cad7f99996e0c6bd7f", "part": "text", "start_byte": 0, "text": "Harry Potter and the Deathly Hallows \u2013 Part 1"}], "table_indices": [], "image_instances": []}]}]}
# st.json(data)

sample_dialogue = data[45]['history']
analysis_results = []

for turn in sample_dialogue:
    question = turn['question']
    answer = turn['answer']
    if answer['type'] == 'image': # for answers of type 'image'
        img_paths = []
        img_dir = r'.\data\final_dataset_images'
        for i in answer['image_instances']: # for each instance in the answer
            img_name = i.get('doc_id', None)  # Ensure 'doc_id' (i.e., the image name) exists
            try:
                img_path = path.join(img_dir, img_name + '.jpg')
            except FileNotFoundError as e:
                img_path = path.join(img_dir, img_name + '.png')
            if path.exists(img_path):
                img_paths.append(img_path)
        for img_path in img_paths:
            pass