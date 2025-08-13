import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path
from PIL import Image
from tqdm import tqdm

from data_loader import load_data, construct_lookups
from clip_analyzer import load_clip


# Define global directories and paths
DATA_DIR = './data/'
QS_PATH = path.join(DATA_DIR, 'MMCoQA_dev.txt')
IMGS_JSONL_PATH = path.join(DATA_DIR, 'multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl')
IMG_FILES_DIR = path.join(DATA_DIR, 'final_dataset_images')

# Modified get_img_txt_similarity to handle NaN returns
def get_img_txt_similarity(img_path: str, text: str, model, processor) -> float:
    """Calculates cosine similarity between an image and a text."""
    try:
        img = Image.open(img_path)
    except Exception:
        return np.nan # Return NaN instead of 0.0 for certain errors
    # Encode both text and image with the wrapped encoders
    inputs = processor(text=[text], images=img, return_tensors="pt", padding=True) # no computation -> no necessity to move to GPU
    # Move inputs to the same device as the model (as the model is moved to GPU if available)
    inputs = {k: v.to(model.device) for k, v in inputs.items()} # keys are 'input_ids', 'attention_mask', 'pixel_values'
    # Disable gradient calculation for inference for efficiency
    with torch.no_grad(): 
        outputs = model(**inputs) # Output keys: ['logits_per_image', 'logits_per_text', 'text_embeds', 'image_embeds', 'text_model_output', 'vision_model_output'] 
    # logits <==> similarity * exp, last_hidden_state <==> prior to NN head, pooler_output <==> after one linear layer (pooling), 
    # Utilize the embeddings for similarity calculation
    img_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    # Normalize (along the dimension of each vector's elements) embeddings for accurate cosine similarity
    normed_img_embed = img_embeds / img_embeds.norm(dim=-1, keepdim=True) # compute Euclidean norm for each vector
    normed_txt_embed = text_embeds / text_embeds.norm(dim=-1, keepdim=True) # make sure the dimensions/shape are the same
    # Calculate similarity using matrix multiplication (i.e., dot product of normalized vectors <==> cosine similarity)
    similarity = (normed_txt_embed @ normed_img_embed.T).item() # extract the scalar values from the tensors
    
    return similarity


if __name__ == "__main__":
    # Load Data and Model
    all_turns = load_data(QS_PATH)
    imgs_lookups = construct_lookups(load_data(IMGS_JSONL_PATH))
    model, processor = load_clip()
    all_scores = []
    print("Computing Q-I similarity scores across the dataset...\n")
    # tqdm for the progress visualization of the loop
    for turn in tqdm(all_turns, desc="Processing Turns"):
        q = turn.get("question")
        # Loop through each image evidence in the answer if there's more than one
        for a in turn.get("answer", []):
            if a.get("modality") == "image":
                for instance in a.get("image_instances", []):
                    img_id = instance.get("doc_id")
                    # In case the evidence's somehow not found
                    if img_id in imgs_lookups:
                        img_filename = imgs_lookups[img_id].get("path")
                        # In case the image evidence failed to load
                        if img_filename:
                            img_path = path.join(IMG_FILES_DIR, img_filename)
                            # Compute its score and add to the list for statistical analysis if valid
                            score = get_img_txt_similarity(img_path, q, model, processor)
                            if not np.isnan(score):
                                all_scores.append(score)
    if not all_scores:
        print("No valid image-question pairs were found or processed -- Please Check Data")
    else:
        scores_array = np.array(all_scores)
        # Compute summary statistics with numpy array methods
        avg_score = np.mean(scores_array)
        median_score = np.median(scores_array)
        max_score = np.max(scores_array)
        min_score = np.min(scores_array)
        std_dev = np.std(scores_array)
        # Check the proportion of scores below 0.40 (upon testing, maximum = 0.382)
        below_threshold = scores_array[scores_array < 0.40]
        proportion_below = (len(below_threshold) / len(scores_array)) * 100
        
        print("**Summary Statistics of Q-I Similarity Scores:**")
        print(f"Total Image Instances Analyzed: {len(all_scores)}")
        print(f"Average Score: {avg_score:.3f}")
        print(f"Median Score: {median_score:.3f}")
        print(f"Maximum Score: {max_score:.3f}")
        print(f"Minimum Score: {min_score:.3f}")
        print(f"Standard Deviation: {std_dev:.3f}")
        print(f"Proportion of scores below 0.30: {proportion_below:.2f}%")

        # Visualize the Distribution
        print("\nGenerating score distribution plot...")
        plt.style.use('seaborn-v0_8-whitegrid') # show grid
        plt.figure(figsize=(12, 6))
        # Utilize pandas' built-in `plot` method to conveniently convert the array to a matplotlib histogram
        pd.Series(scores_array).plot(kind='hist', bins=50, title='Distribution of CLIP Q-I Similarity Scores in MMCoQA Dev Set')
        plt.xlabel("CLIP Similarity Score")
        plt.ylabel("Frequency")
        plt.axvline(avg_score, color='r', linestyle='--', linewidth=2, label=f'Average Score: {avg_score:.2f}')
        plt.axvline(0.4, color='g', linestyle=':', linewidth=2, label='Threshold = 0.40')
        plt.legend()
        plt.tight_layout() # ensure all components of the visualization are properly displayed
        plt.show()