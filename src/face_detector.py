import cv2
import numpy as np

class FaceDetector:
    def __init__(self, cascade_path):
        """
        Initialize face detector using Haar Cascade
        
        Args:
            cascade_path: Path to the Haar Cascade XML file
        """
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
    def detect_faces(self, image):
        """
        Detect faces in the image
        
        Args:
            image: Input image
            
        Returns:
            List of face regions (x, y, w, h)
        """
        # Convert image to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
