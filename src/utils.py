import cv2
import matplotlib.pyplot as plt
import numpy as np

def draw_emotion_results(image, faces, emotions, confidences):
    """
    Draw bounding boxes and emotion labels on image
    
    Args:
        image: Input image
        faces: List of face regions (x, y, w, h)
        emotions: List of emotion labels
        confidences: List of confidence scores
        
    Returns:
        Annotated image
    """
    result_img = image.copy()
    
    for (x, y, w, h), emotion, confidence in zip(faces, emotions, confidences):
        # Draw face bounding box
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Prepare label text
        label = f"{emotion}: {confidence:.2f}"
        
        # Determine text size and position
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(result_img, (x, y - text_size[1] - 10), (x + text_size[0], y), (0, 255, 0), cv2.FILLED)
        
        # Add text
        cv2.putText(result_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return result_img

def get_emotion_color(emotion):
    """
    Get color for emotion visualization
    
    Args:
        emotion: Emotion label
        
    Returns:
        RGB color tuple
    """
    color_map = {
        'Angry': (0, 0, 255),     # Red
        'Disgust': (0, 140, 255), # Orange
        'Fear': (0, 255, 255),    # Yellow
        'Happy': (0, 255, 0),     # Green
        'Sad': (255, 0, 0),       # Blue
        'Surprise': (255, 0, 255),# Purple
        'Neutral': (255, 255, 255)# White
    }
    
    return color_map.get(emotion, (200, 200, 200))
