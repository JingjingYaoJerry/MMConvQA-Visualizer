import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


def load_clip() -> tuple[CLIPModel, CLIPProcessor]:
    """
    Load the CLIP model and processor.

    Args:
        None
    
    Returns:
        tuple(CLIPModel, CLIPProcessor): The CLIP model and processor.
    """
    print("Loading CLIP model and processor...")
    checkpoint = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(checkpoint)
    processor = CLIPProcessor.from_pretrained(checkpoint, use_fast=False) # '_valid_processor_keys'
    return model, processor

def get_img_txt_similarity(img_path: str, text: str, model: CLIPModel, processor: CLIPProcessor) -> float:
    """
    Calculates cosine similarity between an image and a text using a pretrained CLIP model.

    Args:
        image_path (str): Path to the image file.
        text (str): Text to compare with the image.
        model (CLIPModel): Pretrained CLIP model.
        processor (CLIPProcessor): Pretrained CLIP processor.

    Returns:
        similarity (float): Cosine similarity score between the image and text.
    """
    try:
        img = Image.open(img_path)
    except Exception as e:
        print(f"Warning: Could not open image {img_path}. Error: {e}")

        return 0.0
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
    # Test the Loading and similarity calculation
    img_path = r'.\data\final_dataset_images' + r"\f60afcdec9238132fc0f6d11e54c6457" + '.jpg'
    question = "Which Syncopy Inc. movie title(s) that has/have a tall building in the background of its poster?"
    model, processor = load_clip()
    # Get similarity score
    score = get_img_txt_similarity(img_path=img_path, text=question, model=model, processor=processor)
    print(f"Similarity Score: {score}")