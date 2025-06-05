# Gesture Recognition System

This project allows users to control a presentation (such as PowerPoint slides) using hand gestures captured through a webcam. It uses MediaPipe for hand tracking, OpenCV for image processing, and a Random Forest classifier for gesture recognition.

## Project Overview

The project is divided into three key components:

- `collect_gesture_data.py`: Collects hand landmark data using MediaPipe and stores it in a `.pkl` file for training.
- `train_gesture_model.py`: Trains a RandomForestClassifier on the collected data and saves the trained model.
- `recognize_gestures.py`: Uses the trained model to recognize gestures in real-time and trigger slide controls using `pyautogui`.

## Project Structure

gesture-recognition-system/
│
├── collect_gesture_data.py
├── train_gesture_model.py
├── recognize_gestures.py
├── gesture_classifier.pkl
├── gesture_data/ (created after data collection)
│ └── gesture_data.pkl
├── requirements.txt
├── README.md
└── .gitignore



## How to Use

1. Run `collect_gesture_data.py` to collect your gesture samples. You can customize gestures like "open", "next", "stop", etc.
2. Run `train_gesture_model.py` to train the Random Forest model on your collected data.
3. Run `recognize_gestures.py` to start real-time gesture recognition and control your presentation.

### Example Use-Cases:
- Show "open" gesture → Launch slideshow (`F5`)
- Show "next" gesture → Move to next slide
- Show "stop" gesture → Exit slideshow

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt


Main Libraries Used:

    opencv-python

    mediapipe

    scikit-learn

    pyautogui

    numpy


What I Learned

    Hand landmark detection using MediaPipe

    How to collect and structure custom gesture data

    Training and evaluating a Random Forest model

    Real-time application of ML models

    Controlling the system with Python automation



License

This project is for educational and personal use.


---

This version is:
- **Professional**
- **Clear**
- **Shows what you know and did**
- **Easy for anyone to understand**

Would you like me to prepare a `requirements.txt` or `.gitignore` for this project as well?
