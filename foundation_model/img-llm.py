import pickle
import torch
import pandas as pd
from transformers import AutoProcessor, AutoModelForImageTextToText

# Vision Llama model
def load_model(model_name="meta-llama/Llama-3.2-11B-Vision"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(model_name).to(device)
    return processor, model, device

# Load the preprocessed image data
def load_pkl_images(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return [item['img_array'] for _, item in data.iterrows()]

# Load the 'y' file image labels
def load_pkl_labels(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return [item['label'] for _, item in data.iterrows()]

# Processes images and generate descriptions
def process_images_with_vision_llama(images, processor, model, device):
    results = []
    for img in images:
        inputs = processor(images=img, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_length=50)
        caption = processor.batch_decode(output, skip_special_tokens=True)[0]
        results.append(caption)
    return results

# Train and evaluate the model...to see if it can generate good descriptions
def train_and_evaluate(train_pkl, train_labels_pkl, test_pkl, test_labels_pkl):
    processor, model, device = load_model()
    
    print("Loading training data...")
    train_images = load_pkl_images(train_pkl)
    train_labels = load_pkl_labels(train_labels_pkl)
    
    print("Processing training images...")
    train_descriptions = process_images_with_vision_llama(train_images, processor, model, device)
    
    print("Loading testing data...")
    test_images = load_pkl_images(test_pkl)
    test_labels = load_pkl_labels(test_labels_pkl)
    
    print("Processing testing images...")
    test_descriptions = process_images_with_vision_llama(test_images, processor, model, device)
    
    print("Training Data Descriptions:")
    for i, (desc, label) in enumerate(zip(train_descriptions, train_labels)):
        print(f"Train Image {i+1}: {desc} | Label: {label}")
    
    print("\nTesting Data Descriptions:")
    for i, (desc, label) in enumerate(zip(test_descriptions, test_labels)):
        print(f"Test Image {i+1}: {desc} | Label: {label}")

if __name__ == "__main__":
    train_pkl_file = "train_mri_meta.pkl"  # Replace with our img training data path
    train_labels_pkl_file = "train_labels.pkl"  # Replace with our training labels path
    test_pkl_file = "test_mri_meta.pkl"  # Replace with our img testing data path
    test_labels_pkl_file = "test_labels.pkl"  # Replace with our testing labels path
    
    train_and_evaluate(train_pkl_file, train_labels_pkl_file, test_pkl_file, test_labels_pkl_file)