# utils.py
import pickle
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Embedding, Dropout, Concatenate, Layer, Reshape, Bidirectional
) 

def idx_to_word(tokenizer, index):
    """
    Converts a token index back to the corresponding word using tokenizer.
    """
    return tokenizer.index_word.get(index, None)

def load_tokenizer(path):
    """
    Loads a pickled tokenizer from the specified path.
    """
    with open(path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

class BahdanauAttention(Layer):
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = tf.reduce_sum(attention_weights * features, axis=1)
        return context_vector

    def get_config(self):
        config = super(BahdanauAttention, self).get_config()
        config.update({"units": self.W1.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_image_with_caption(image_path, caption):
    """
    Plots an image with the corresponding caption.

    Args:
    - image_path (str): Path to the image.
    - caption (str): Generated caption for the image.
    """
    # Read and plot the image
    img = mpimg.imread(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')  # Turn off axis
    plt.title(caption, fontsize=12)
    plt.show()
