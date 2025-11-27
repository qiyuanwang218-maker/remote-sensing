import json
from tqdm import tqdm
import open_clip
import torch
import os
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_image_data(image_data_path):
    """Load image metadata (image_id and file names) from the dataset."""
    with open(image_data_path, 'r') as file:
        data = json.load(file)['images']
    images = [{'image_id': item['imgid'], 'file_name': item['filename']} for item in data]
    return images

def load_noun_classes(noun_classes_path):
    """Load noun classes from noun_classes.json."""
    with open(noun_classes_path, 'r') as file:
        noun_classes = json.load(file)
    categories = [item[0] for item in noun_classes]  # Extract the class names
    return categories

def encode_text_classes(categories, model, tokenizer, device):
    """Encode text categories using OpenCLIP."""
    text_prompts = [f"a photo of a {category}" for category in categories]
    with torch.no_grad():
        text_inputs = tokenizer(text_prompts).to(device)  # Tokenize text
        text_features = model.encode_text(text_inputs).cpu().numpy()

    # L2 normalization of text features
    text_features = text_features / np.linalg.norm(text_features, axis=-1, keepdims=True)

    return text_features

def encode_images(images, image_path, model, preprocess, device):
    """Encode images into feature vectors using OpenCLIP."""
    image_ids = [img['image_id'] for img in images]
    bs = 64  # Batch size
    image_features = []

    for idx in tqdm(range(0, len(images), bs), desc="Encoding images"):
        image_batch = [
            preprocess(Image.open(os.path.join(image_path, img['file_name'])).convert("RGB"))
            for img in images[idx:idx + bs]
        ]
        image_input = torch.stack(image_batch).to(device)
        with torch.no_grad():
            image_features.append(model.encode_image(image_input).cpu().numpy())

    image_features = np.concatenate(image_features)

    # L2 normalization of image features
    image_features = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)

    return image_ids, image_features

def predict_image_classes(image_features, text_features, categories, threshold=0.2):
    """Predict the most likely class for each image."""
    similarity = np.dot(image_features, text_features.T)  # Compute cosine similarity
    predictions = []

    for sim in similarity:
        pred = [
            {"category": categories[i], "score": float(sim[i])}
            for i in range(len(categories)) if sim[i] > threshold
        ]
        pred = sorted(pred, key=lambda x: x["score"], reverse=True)  # Sort by score
        predictions.append(pred)

    return predictions

def save_predictions(image_ids, predictions, output_path):
    """Save predictions to a JSON file."""
    results = {str(image_id): pred for image_id, pred in zip(image_ids, predictions)}
    with open(output_path, 'w') as file:
        json.dump(results, file, indent=4)

def main():
    image_data_path = "data/dataset_coco.json"  # Path to dataset metadata
    image_path = "data/images/"  # Path to image files
    noun_classes_path = "data/noun_classes.json"  # Path to noun classes file
    output_path = "data/retrieved_class.json"  # Path to save predictions
    threshold = 0.3

    print("Loading data...")
    images = load_image_data(image_data_path)
    categories = load_noun_classes(noun_classes_path)

    print("Initializing OpenCLIP...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model_name = 'ViT-B-32'  # Specify model architecture
    clip_model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name)
    tokenizer = open_clip.get_tokenizer(clip_model_name)

    # Load pre-trained weights
    path_to_your_checkpoints = '/root/work/checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38'
    ckpt = torch.load(f"{path_to_your_checkpoints}/RemoteCLIP-{clip_model_name}.pt", map_location="cpu")
    clip_model.load_state_dict(ckpt)
    clip_model = clip_model.to(device).eval()

    print("Encoding text categories...")
    text_features = encode_text_classes(categories, clip_model, tokenizer, device)

    print("Encoding images...")
    image_ids, image_features = encode_images(images, image_path, clip_model, preprocess, device)

    print("Predicting image classes...")
    predictions = predict_image_classes(image_features, text_features, categories, threshold=threshold)

    print(f"Saving predictions to {output_path}...")
    save_predictions(image_ids, predictions, output_path)
    print("Done!")

if __name__ == "__main__":
    main()
