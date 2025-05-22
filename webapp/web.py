from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import sys
import tensorflow as tf

# Insert project root (one level up) into Python’s module search path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
    
from app.preprocess import load_and_preprocess_image
from app.feature_extractor import build_feature_extractor
from app.inference import generate_caption
from app.speech import text_to_speech
from app.utils import BahdanauAttention
from tensorflow.keras.models import load_model
import pickle

# 2) Build the path to the .keras file in data/
MODEL_PATH = os.path.join(BASE_DIR, "data", "best_caption_model.keras")
TOKENISER_PATH = os.path.join(BASE_DIR, "data", "tokenizer.pkl")

app = Flask(__name__)

# Setup configuration and constants
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_LENGTH = 34
#os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=False)

# Load model and tokenizer once during application startup
model=tf.keras.models.load_model(MODEL_PATH, custom_objects={'BahdanauAttention': BahdanauAttention, 'KerasLayer': tf.keras.layers.Layer}, compile=False)
fe_model = build_feature_extractor()
with open(TOKENISER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = None
    image_path = None

    if request.method == 'POST':
        img = request.files.get('image')

        if img and allowed_file(img.filename):
            filename = secure_filename(img.filename)
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            img.save(image_path)

            try:
                # Process the uploaded image and generate caption
                processed_image = load_and_preprocess_image(image_path)
                feature = fe_model.predict(processed_image)  # Feature extraction
                caption = generate_caption(model, tokenizer, feature, MAX_LENGTH)
                
                # Convert caption to speech
                text_to_speech(caption)

            except Exception as e:
                # Handle exceptions and provide feedback to the user
                print(f"Error occurred: {e}")
                caption = "There was an error processing the image. Please try again."

        else:
            caption = "Invalid file type. Please upload an image file."

    # Render the result page with the caption and image
    return render_template('index.html', image_path=image_path, caption=caption)

if __name__ == '__main__':
    app.run(debug=True)
