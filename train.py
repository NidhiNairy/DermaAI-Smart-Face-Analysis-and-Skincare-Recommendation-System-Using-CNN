"""
Utility functions for model training and evaluation
"""
import numpy as np
import os
from tensorflow import keras

def prepare_training_data(data_dir):
    """
    Prepare training data from directory structure
    
    Expected structure:
    data_dir/
        skin_type/
            oily/
            dry/
            ...
        skin_tone/
            light/
            medium/
            ...
        skin_concerns/
            acne/
            pigmentation/
            ...
    """
    # This is a placeholder for actual data preparation
    # In production, you would load and preprocess images from directories
    pass

def save_model_checkpoint(model, epoch, loss, filepath):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)
    print(f"Model checkpoint saved: Epoch {epoch}, Loss: {loss:.4f}")

def load_model_checkpoint(filepath):
    """Load model checkpoint"""
    if os.path.exists(filepath):
        return keras.models.load_model(filepath)
    return None


