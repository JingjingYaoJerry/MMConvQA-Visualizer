# MMConvQA Visualizer

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.47+-red.svg)](https://streamlit.io)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![Made-with-Love](https://img.shields.io/badge/Made%20with-❤️-ff69b4.svg)](https://www.linkedin.com/in/%E4%BA%AC%E6%99%B6-%E5%A7%9A-9997b5180/)

An interactive tool for the exploration on the `MMConvQA` dataset, built to better understand its multimodal, and conversational structure, with multi-answer/multi-evidence support and text-image similarity analysis via `clip-vit-large-patch14`.

---

### Video Demonstration

Due to the use-case restrictions of the `openai/clip-vit-large-patch14` model checkpoint, this application is intended for local research demonstration only and hence not publicly deployed.

**Instead, a short video walkthrough of the application is available here:**

**[Link to the video]**

![Table Evidence Construction Demo](/assets/images/table_demo.png "Table Evidence Construction Demo")

![Multimodal Support Demo](/assets/images/multimodal_demo.png "Multimodal Support Demo")

![Multi-Evidence Support Demo](/assets/images/multi-evidence_demo.png "Multi-Evidence Support Demo")

![Q-I Similarity Analysis Demo](/assets/images/similarity_demo.png "Q-I Similarity Analysis Demo")

---

### Key Features

* **Conversation Reconstruction**: Parsing the flat `MMCoQA_dev.txt` list and grouping individual questions into chronological conversations according to their `qid`.
* **Efficient Data Handling**: Constructing in-memory lookup index for all evidence modalities, ensuring a responsive user experience.
* **Multimodal Evidence Visualization**: Dynamically displaying evidence for each conversational turn, appropriately handling:
    * **Images**: Rendering images directly from the local storage.
    * **Tables**: Reconstructing tables from the source `...tables.jsonl` file, highlighting the specific cells cited as evidence.
    * **Text**: Showing the full text passage cited as evidence.
* **Multi-Answer & Multi-Evidence Supporting**: Appropriately handling cases where a single question has multiple answers, or an answer is supported by multiple multimodal evidences.
* **Interactive Q-I Analysis**: Integrating the `clip-vit-large-patch14` CLIP model to provide naive similarity scores between a question and its associated image evidence, offering insights into text-image alignment.
* **Image Evidences' Ranking**: Rank all image evidence instances in the current conversation by their CLIP similarity to the question, highlighting the position of the correct evidence image.

--- 

### Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/JingjingYaoJerry/MMConvQA-Visualizer.git
    ```

2.  **Set up the data directory:**
    * Create a `data` folder inside the root.
    * Download the official `MMCoQA` dataset files and place them inside `data/`. The final structure should be something like:
        ```
        MMConvQA-Visualizer/
        ├── data/
        │   ├── MMCoQA_dev.txt
        │   ├── MMCoQA_test.txt
        │   ├── MMCoQA_train.txt
        │   ├── qrels.txt
        │   ├── multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl
        │   ├── multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl
        │   ├── multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl
        │   └── final_dataset_images/
        │       ├── ... (all .jpg and .png files)
        └── ... (.py files)
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

---

### Project Structure

* `app.py`: The Streamlit application file, handling UI with interactions.
* `data_loader.py`: The module for loading, parsing, and pre-processing the MMConvQA data into groups, tables and efficient lookup structures.
* `clip_analyzer.py`: The module for loading the CLIP model via Hugging Face and performing similarity analysis.
* `requirements.txt`: A list of all necessary Python packages.
* `./data/`: The directory for storing all `MMCoQA` datasets ([please refer to the team's project page](https://github.com/liyongqi67/MMCoQA?tab=readme-ov-file)).

---

### Acknowledgements

This project is built upon the public dataset from the paper, and the foundational work from the team:

> Yongqi Li, Wenjie Li, and Liqiang Nie. 2022. **MMCoQA: Conversational Question Answering over Text, Tables, and Images.** *In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4220-4231.*