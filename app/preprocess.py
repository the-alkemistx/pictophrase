# app/preprocess.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input

def load_and_preprocess_image(image_path):
    pil = load_img(image_path, target_size=(299, 299))
    arr = img_to_array(pil)
    arr = preprocess_input(arr)
    return tf.expand_dims(arr, axis=0)  # (1, 299, 299, 3)
