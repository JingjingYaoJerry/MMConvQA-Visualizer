from transformers import pipeline

# Utilizing the Hugging Face Pipeline and CLIP model for the image classification task with no example
pipe = pipeline("zero-shot-image-classification", model="openai/clip-vit-large-patch14")

def get_similarity(img_path: str, labels: list[str]) -> float:
    """Get the similarity score for an image according to the model's card."""
    scores = pipe(img_path, candidate_labels=labels)
    # print(f"Scores: {scores}")
    # scores = [{'score': float, 'label': str}, ...]
    return scores[0]['score'] # get the first-ranked highest score

if __name__ == "__main__":
    # Example usage from the model's card
    img_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png"
    labels = ["animals", "humans", "landscape"] # candidate labels
    similarity = get_similarity(img_path, labels)
    print(f"Similarity Score: {similarity}")