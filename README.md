# Emotion-Recognition
# Emotion Detection Project

This project uses OpenCV and deep learning to detect faces in images or video streams and classify their emotions.

## Features
- Face detection using Haar Cascade classifiers
- Emotion classification (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- Support for both image and video processing

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Make sure you have the required models:
   - The Haar Cascade model will be downloaded automatically
   - For the emotion classifier, you'll need to:
     - Use the placeholder model (for demo purposes only)
     - OR train your own model
     - OR download a pre-trained model

## Usage

Process an image:
```
python main.py --image path/to/image.jpg
```

Process video from webcam:
```
python main.py
```

Process a video file:
```
python main.py --video path/to/video.mp4
```
