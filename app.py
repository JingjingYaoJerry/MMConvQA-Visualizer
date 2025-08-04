import streamlit as st
import pandas as pd
from os import path
from PIL import Image

from data_loader import prepare_all_data, construct_table_from_lookups
from clip_analyzer import load_clip, get_img_txt_similarity


# Streamlit configs & title
st.set_page_config(page_title='MMCoQA Explorer', layout='wide', initial_sidebar_state='auto')
st.title("MMCoQA Explorer")
st.info("A tool for the exploration on MMCoQA data with multi-answer and multi-evidence support.")
st.divider() # Separator line

# Define global directories and paths
DATA_DIR = './data/'
QS_PATH = path.join(DATA_DIR, 'MMCoQA_dev.txt')
IMGS_JSONL_PATH = path.join(DATA_DIR, 'multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl')
IMG_FILES_DIR = path.join(DATA_DIR, 'final_dataset_images')
TABS_PATH = path.join(DATA_DIR, 'multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl')
TXTS_PATH = path.join(DATA_DIR, 'multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl')

# Define helper function to display multi-modal evidences
def display_evidence_card(question: str, ans: dict, turn_qid: str, card_index: int) -> None:
    """Create a self-contained card for each answer and all its evidences."""
    modality = ans.get('modality')
    with st.container(height=None, border=True, key=f"card_{turn_qid}_{card_index}"):
        st.subheader(f"Answer: \"{ans['answer']}\"")
        st.caption(f"Modality: {modality.upper()}")
        if modality == 'image' and ans.get('image_instances'):
            # Display the modality in bold
            st.markdown("**Image Evidence Instances:**")
            for i, instance in enumerate(ans['image_instances']):
                # Retrieve the image evidence by its ID provided in the answer
                img_id = instance['doc_id']
                # In case the evidence's somehow not found
                if img_id in imgs_lookups:
                    img_filename = imgs_lookups[img_id].get('path')
                    img_path = path.join(IMG_FILES_DIR, img_filename)
                    # In case the image evidence failed to load
                    if path.exists(img_path):
                        # Display the image evidence with its title, image content, analysis and URL
                        with st.expander(label=f"Instance {i+1} '{imgs_lookups[img_id].get('title', '?')}'"):
                            st.image(img_path, caption=f"Instance {i+1} '{imgs_lookups[img_id].get('title', '?')}'", use_column_width=True)
                            # Unique key for each button is crucial for Streamlit
                            if st.button("Analyze Q-I Similarity", key=f"clip_{turn_qid}_{card_index}_{i}"):
                                with st.spinner("Running CLIP..."):
                                    score = get_img_txt_similarity(img_path, question, model, processor)
                                    st.metric(label="CLIP Score", value=f"{score:.2f}")
                    else: st.warning(f"Image file `{img_filename}` not found.")
                else: st.warning(f"Image instance with ID `{img_id}` cannot be presented.")
        elif modality == 'table' and ans.get('table_indices'):
            # Display the modality in bold
            st.markdown("**Table Evidence Instances:**")
            # Unlike image and text evidences, table_id is provided in the "question level"
            # and every evidence is provided in indices in the "answer level"
            # Hence, get back to the "question level" to get the table ID first
            tab_id_from_q = st.session_state.current_q.get('table_id')
            # In case the evidence's somehow not found
            if tab_id_from_q in tabs_lookups and tabs_lookups[tab_id_from_q].get('table'):
                tab_i = tabs_lookups[tab_id_from_q].get('table')
                df = construct_table_from_lookups(tab_i)
                # Highlight cells based on the indices
                def highlight_cells(row): # function to be applied
                    styles = [''] * len(row) # as for each row
                    # For each pair of indices in the answer
                    for row_idx, col_idx in ans['table_indices']:
                        # If the indices are valid, highlight the cell
                        if row.name == row_idx and col_idx < len(row):
                            styles[col_idx] = 'background-color: #FFFF00;' # common yellow highlight
                    return styles
                # Display the table evidence with its title, table content and URL
                with st.expander(label=f"Table Evidence from '{tabs_lookups[tab_id_from_q].get('title', '?')}'"):
                    st.dataframe(df.style.apply(highlight_cells, axis=1), use_container_width=True) # for each row
                    st.page_link(page=tabs_lookups[tab_id_from_q].get('url'), label=tabs_lookups[tab_id_from_q].get('url'))
            else: st.warning(f"Table instance with ID `{tab_id_from_q}` cannot be presented.")
        elif modality == 'text' and ans.get('text_instances'):
            # Display the modality in bold
            st.markdown("**Text Evidence Instances:**")
            # For each text evidence in one answer
            for i, instance in enumerate(ans['text_instances']):
                # Retrieve the text evidence by its ID provided in the answer
                txt_id = instance['doc_id']
                # In case the evidence's somehow not found
                if txt_id in txts_lookups:
                    txt_i = txts_lookups[txt_id]
                    # Display the text evidence with its title, content and URL
                    with st.expander(label=f"Instance {i+1} from '{txt_i.get('title', '?')}'"):
                        st.write(txt_i['text'])
                        st.page_link(page=txt_i.get('url'), label=txt_i.get('url'))
                else: st.warning(f"Text instance with ID `{txt_id}` cannot be presented.")
        else:
            st.write("No specific evidence instances found for this answer.")

# Load and cache the data to avoid reloading on every interaction
@st.cache_data
def cache_prepared() -> tuple:
    """Cache all processed conversations & lookups."""
    return prepare_all_data(QS_PATH, IMGS_JSONL_PATH, TABS_PATH, TXTS_PATH)

@st.cache_resource
def cache_clip() -> tuple:
    """Cache the CLIP model in Streamlit cache for 'resource' according to the documentation."""
    return load_clip()

try:
    convs, imgs_lookups, tabs_lookups, txts_lookups = cache_prepared()
    model, processor = cache_clip()
except Exception as e:
    st.error(f"Error loading data or model: {e}")
    st.stop()

# Sidebar for conversation selection
st.sidebar.header("Conversation Selection")
conv_ids = sorted(convs.keys())
selected_conv_id = st.sidebar.selectbox("Choose a Conversation: ", conv_ids)
selected_conv = convs[selected_conv_id]

# Main Content Display
if selected_conv_id:
    st.header(f"MMCoQA Exploring on Conversation `{selected_conv_id}`")
    # For each turn/question in one conversation
    for i, turn in enumerate(selected_conv):
        # Store the turn state for later access
        st.session_state.current_q = turn
        with st.container(border=True):
            st.markdown(f"## Turn {i+1}: `{turn['qid']}`")
            # Display the current question for referencing
            st.markdown(f"### Question: {turn['question']}")
            # For each answer in one turn/question
            for j, ans in enumerate(turn['answer']):
                display_evidence_card(turn['question'], ans, turn['qid'], j)
else:
    st.header("MMCoQA Explorer")