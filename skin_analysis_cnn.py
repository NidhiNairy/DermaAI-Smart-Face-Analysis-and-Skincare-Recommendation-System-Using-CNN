"""
CNN Model for skin analysis (skin type, skin tone, skin concerns)
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

class SkinAnalysisCNN:
    """CNN model for analyzing skin characteristics"""
    
    def __init__(self, model_path=None):
        self.model = None
        if model_path is None:
            # Get the directory where this file is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_path = os.path.join(current_dir, 'saved_models', 'skin_analysis_model.h5')
        else:
            self.model_path = model_path
        self.load_or_create_model()
        
        # Skin type classes
        self.skin_types = ['oily', 'dry', 'combination', 'sensitive', 'normal']
        
        # Skin tone classes
        self.skin_tones = ['light', 'medium', 'tan', 'dark', 'deep']
        
        # Skin concern classes
        self.skin_concerns = [
            'acne', 'pigmentation', 'dark_circles', 'dark_spots',
            'eye_bags', 'open_pores', 'blackheads', 'whiteheads',
            'wrinkles', 'fine_lines', 'redness', 'dullness'
        ]
    
    def load_or_create_model(self):
        """Load existing model or create a new one"""
        if os.path.exists(self.model_path):
            try:
                self.model = keras.models.load_model(self.model_path)
                print(f"Loaded model from {self.model_path}")
            except:
                print("Could not load existing model. Creating new model...")
                self.create_model()
        else:
            print("No existing model found. Creating new model...")
            self.create_model()
    
    def create_model(self):
        """Create CNN model architecture"""
        # Input layer
        inputs = keras.Input(shape=(224, 224, 3))
        
        # Base CNN architecture (using transfer learning approach)
        # Feature extraction layers
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Multi-task output heads
        # Skin type classification
        skin_type_output = layers.Dense(len(self.skin_types), activation='softmax', name='skin_type')(x)
        
        # Skin tone classification
        skin_tone_output = layers.Dense(len(self.skin_tones), activation='softmax', name='skin_tone')(x)
        
        # Skin concerns (multi-label classification)
        skin_concerns_output = layers.Dense(len(self.skin_concerns), activation='sigmoid', name='skin_concerns')(x)
        
        # Create model
        self.model = keras.Model(
            inputs=inputs,
            outputs=[skin_type_output, skin_tone_output, skin_concerns_output],
            name='skin_analysis_cnn'
        )
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'skin_type': 'categorical_crossentropy',
                'skin_tone': 'categorical_crossentropy',
                'skin_concerns': 'binary_crossentropy'
            },
            metrics={
                'skin_type': 'accuracy',
                'skin_tone': 'accuracy',
                'skin_concerns': 'accuracy'
            }
        )
        
        print("Created new CNN model")
    
    def analyze_skin(self, preprocessed_image):
        """
        Analyze skin characteristics from preprocessed image
        
        Args:
            preprocessed_image: Preprocessed image array (1, 224, 224, 3)
            
        Returns:
            Dictionary with analysis results
        """
        if self.model is None:
            # Return default predictions if model not loaded
            return self._get_default_predictions()
        
        try:
            # Make predictions
            predictions = self.model.predict(preprocessed_image, verbose=0)
            
            skin_type_pred, skin_tone_pred, skin_concerns_pred = predictions
            
            # Get skin type
            skin_type_idx = np.argmax(skin_type_pred[0])
            skin_type = self.skin_types[skin_type_idx]
            skin_type_confidence = float(skin_type_pred[0][skin_type_idx])
            
            # Get skin tone
            skin_tone_idx = np.argmax(skin_tone_pred[0])
            skin_tone = self.skin_tones[skin_tone_idx]
            skin_tone_confidence = float(skin_tone_pred[0][skin_tone_idx])
            
            # Get skin concerns (threshold: 0.5)
            concerns_threshold = 0.5
            detected_concerns = []
            concerns_confidences = {}
            
            for i, concern in enumerate(self.skin_concerns):
                confidence = float(skin_concerns_pred[0][i])
                if confidence >= concerns_threshold:
                    detected_concerns.append(concern)
                    concerns_confidences[concern] = confidence
            
            return {
                'skin_type': skin_type,
                'skin_tone': skin_tone,
                'skin_concerns': detected_concerns,
                'confidence': {
                    'skin_type': skin_type_confidence,
                    'skin_tone': skin_tone_confidence,
                    'skin_concerns': concerns_confidences
                }
            }
        
        except Exception as e:
            print(f"Error in skin analysis: {e}")
            return self._get_default_predictions()
    
    def _get_default_predictions(self):
        """Return default predictions when model is not available"""
        return {
            'skin_type': 'normal',
            'skin_tone': 'medium',
            'skin_concerns': [],
            'confidence': {
                'skin_type': 0.5,
                'skin_tone': 0.5,
                'skin_concerns': {}
            }
        }
    
    def save_model(self, path=None):
        """Save the model to disk"""
        save_path = path or self.model_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        print(f"Model saved to {save_path}")

