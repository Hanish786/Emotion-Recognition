import os
import cv2
import numpy as np
import tensorflow as tf
from src.face_detector import FaceDetector
from src.emotion_classifier import EmotionClassifier
from src.utils import draw_emotion_results

def process_image(image_path):
    """
    Process a single image for emotion detection
    
    Args:
        image_path: Path to the image file
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Initialize face detector
    face_detector = FaceDetector("models/haarcascade_frontalface_default.xml")
    
    # Detect faces
    faces = face_detector.detect_faces(image)
    
    # If no faces detected
    if len(faces) == 0:
        print("No faces detected in the image.")
        return
    
    # Initialize emotion classifier
    classifier = EmotionClassifier("models/emotion_model.h5")
    
    # Process each face
    emotions = []
    confidences = []
    
    for (x, y, w, h) in faces:
        # Extract face region
        face_img = image[y:y+h, x:x+w]
        
        # Predict emotion
        emotion, confidence = classifier.predict_emotion(face_img)
        
        # Store results
        emotions.append(emotion)
        confidences.append(confidence)
    
    # Draw results on image
    result_img = draw_emotion_results(image, faces, emotions, confidences)
    
    # Display results
    cv2.imshow("Emotion Detection", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_source=0):
    """
    Process video for emotion detection
    
    Args:
        video_source: Camera index or video file path
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return
    
    # Initialize face detector
    face_detector = FaceDetector("models/haarcascade_frontalface_default.xml")
    
    # Initialize emotion classifier
    classifier = EmotionClassifier("models/emotion_model.h5")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        faces = face_detector.detect_faces(frame)
        
        # Process each face
        emotions = []
        confidences = []
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            
            # Skip if face region is empty
            if face_img.size == 0:
                continue
                
            # Predict emotion
            try:
                emotion, confidence = classifier.predict_emotion(face_img)
                
                # Store results
                emotions.append(emotion)
                confidences.append(confidence)
            except Exception as e:
                print(f"Error processing face: {e}")
                emotions.append("Unknown")
                confidences.append(0.0)
        
        # Draw results on frame
        result_frame = draw_emotion_results(frame, faces, emotions, confidences)
        
        # Display result
        cv2.imshow("Emotion Detection", result_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    """
    Main function with options to process image or video
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Emotion Detection with OpenCV")
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--video", help="Path to video file, or set to 'cam' for webcam", default="cam")
    
    args = parser.parse_args()
    
    if args.image:
        process_image(args.image)
    else:
        video_source = 0 if args.video == "cam" else args.video
        process_video(video_source)

if __name__ == "__main__":
    # Download or create necessary models
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Download Haar Cascade if not exists
    cascade_path = "models/haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        print("Downloading Haar Cascade model...")
        import urllib.request
        haar_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(haar_url, cascade_path)
    
    
    emotion_model_path = "models/emotion_model.h5"
    if not os.path.exists(emotion_model_path):
        print("WARNING: Emotion detection model not found at", emotion_model_path)
        print("You'll need to train or download a model before running this program.")
        print("For this example, we'll create a placeholder model.")
        
        # Create a simple placeholder model (not for actual use)
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
        
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')  # 7 emotions
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.save(emotion_model_path)
        print(f"Created placeholder model at {emotion_model_path}")
    
    main()
