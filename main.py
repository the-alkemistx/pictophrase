import pickle
import numpy as np
import tensorflow as tf
import keras
import os

from app.preprocess import load_and_preprocess_image
from app.feature_extractor import build_feature_extractor
from app.inference import generate_caption
from app.caption_utils import post_process_caption
from app.speech import text_to_speech
from app.utils import BahdanauAttention, plot_image_with_caption

# ====================
# Configuration
# ====================
IMAGE_DIR = 'data/sample_images'  # Directory of images to be processed
MODEL_PATH = 'data/best_caption_model.keras'
TOKENIZER_PATH = 'data/tokenizer.pkl'
MAX_LENGTH = 34
VOCAB_SIZE = 6000 #if max_lenth=40 then proportionally increase vocab size to 6000, if max_length=50 then vocab size=8000, if max_length=60 then vocab size=10000

# ====================
# Setup
# ====================
keras.config.enable_unsafe_deserialization()

def load_tokenizer(tokenizer_path):
    """Load the tokenizer from a pickle file."""
    with open(tokenizer_path, 'rb') as f:
        return pickle.load(f)

def load_caption_model(model_path):
    """Load the caption generation model."""
    return tf.keras.models.load_model(MODEL_PATH, custom_objects={'BahdanauAttention': BahdanauAttention, 'KerasLayer': tf.keras.layers.Layer}, compile=False)

def process_image_directory(image_dir):
    """Process all images in the directory."""
    print(f"🧹 [Step 2] Preprocessing images in directory: {image_dir}...")
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("❌ No images found in the directory.")
        return []

    photos = []
    image_paths = []
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image_paths.append(image_path)  # Store image path for plotting
        processed_image = load_and_preprocess_image(image_path)  # Preprocess each image
        photos.append(processed_image)

    return photos, image_paths

# ====================
# Main Pipeline
# ====================
def main():
    print("\n📥 [Step 1] Loading model and tokenizer...")
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    model = load_caption_model(MODEL_PATH)
    model._name = 'caption_model'
    print("✅ Model loaded successfully.")

    # Process images from the directory
    photos, image_paths = process_image_directory(IMAGE_DIR)

    if not photos:
        return

    # Extract features using the feature extractor
    feature_extractor = build_feature_extractor()

    for idx, (photo, image_path) in enumerate(zip(photos, image_paths)):
        print(f"\n🧠 [Step 3] Extracting features for image {idx + 1}...")
        photo_features = feature_extractor(photo)

        print("\n📝 [Step 4] Generating caption...")
        raw_caption = generate_caption(model, tokenizer, photo_features, max_length=MAX_LENGTH)
        print(f"🗒️ Raw Caption for image {idx + 1}: {raw_caption}")

        final_caption = post_process_caption(raw_caption)
        print(f"✅ Final Caption for image {idx + 1}: {final_caption}")

        print("\n🔊 [Step 5] Converting caption to speech...")
        text_to_speech(final_caption)

        # Plot the image with its caption
        plot_image_with_caption(image_path, final_caption)

if __name__ == "__main__":
    main()
