# model.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Embedding, Dropout,
    RepeatVector, Concatenate, Activation, Multiply, Softmax
)

def build_caption_model(vocab_size, max_length):
    """
    Builds an image captioning model with a simple attention-like mechanism.
    """

    # --- Image Feature Encoder ---
    image_input = Input(shape=(2048,), name='image_input')
    image_dense = Dense(256, activation='relu', name='image_dense')(image_input)
    image_repeat = RepeatVector(max_length, name='image_repeat')(image_dense)

    # --- Text Input Encoder ---
    caption_input = Input(shape=(max_length,), name='caption_input')
    caption_embed = Embedding(
        input_dim=vocab_size,
        output_dim=256,
        mask_zero=True,
        name='caption_embedding'
    )(caption_input)
    caption_lstm = LSTM(256, return_sequences=True, name='caption_lstm')(caption_embed)

    # --- Fusion Layer ---
    merged = Concatenate(axis=-1, name='concat_features')([image_repeat, caption_lstm])

    # --- Attention Mechanism ---
    attention_dense = Dense(1, activation='tanh', name='attention_dense')(merged)
    attention_weights = Softmax(axis=1, name='attention_softmax')(attention_dense)

    context = Multiply(name='context_multiply')([attention_weights, caption_lstm])

    # --- Final Caption Decoder ---
    context_lstm = LSTM(256, name='context_lstm')(context)
    output_dense = Dense(256, activation='relu', name='output_dense')(context_lstm)
    output = Dense(vocab_size, activation='softmax', name='output')(output_dense)

    model = Model(inputs=[image_input, caption_input], outputs=output, name="caption_model")

    return model
