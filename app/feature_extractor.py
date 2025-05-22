# app/feature_extractor.py

from tensorflow.keras.applications.inception_v3 import InceptionV3

def build_feature_extractor():
    """
    Returns a frozen InceptionV3 model that maps
    (batch_size, 299, 299, 3) → (batch_size, 2048).
    
    Usage:
        extractor = build_feature_extractor()
        features  = extractor(preprocessed_batch)  # a tf.Tensor
    """
    model = InceptionV3(include_top=False,
                        pooling='avg',
                        weights='imagenet')
    model.trainable = False
    return model

# Alias for backwards compatibility
get_feature_extractor = build_feature_extractor
