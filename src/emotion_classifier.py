import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class EmotionClassifier:
    def __init__(self, model_path):
        """
        Initialize emotion classifier
        
        Args:
            model_path: Path to pre-trained emotion detection model
        """
        self.model = load_model(model_path)
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
    def preprocess_face(self, face_img):
        """
        Preprocess face image for model input
        
        Args:
            face_img: Face image
            
        Returns:
            Preprocessed face image
        """
        # Resize to model input size
        face_img = cv2.resize(face_img, (48, 48))
        
        # Convert to grayscale if not already
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Normalize pixel values
        face_img = face_img / 255.0
        
        face_img = np.expand_dims(face_img, axis=-1)
        face_img = np.expand_dims(face_img, axis=0)
        
        return face_img
        
    def predict_emotion(self, face_img):
        """
        Predict emotion from face image
        
        Args:
            face_img: Face image
            
        Returns:
            Predicted emotion label and confidence score
        """
        # Preprocess face
        processed_img = self.preprocess_face(face_img)
        
        # Make prediction
        prediction = self.model.predict(processed_img)[0]
        
        # Get emotion label and confidence
        emotion_idx = np.argmax(prediction)
        emotion_label = self.emotions[emotion_idx]
        confidence = prediction[emotion_idx]
        
        return emotion_label, confidence
