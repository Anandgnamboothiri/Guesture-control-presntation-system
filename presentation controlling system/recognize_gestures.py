import cv2
import numpy as np
import mediapipe as mp
import pickle
import pyautogui  # For keyboard control
import time

# Prompt user for dataset location
dataset_path = r"D:\presentation controlling system\gesture_classifier.pkl"

# Load the trained gesture classifier
with open(dataset_path, 'rb') as f:
    clf = pickle.load(f)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

last_recognition_time = time.time()
recognition_interval = 3  # 3 seconds interval

# Initialize the hands module
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    print("Gesture Recognition for Slideshow:")
    print(" - 'open': Start slideshow")
    print(" - 'next': Next slide")
    print(" - 'stop': Exit slideshow")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip the frame horizontally for natural interaction
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB (required by MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hand landmarks
        results = hands.process(frame_rgb)

        # Check if the required time has passed since last recognition
        if time.time() - last_recognition_time >= recognition_interval:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Extract hand landmarks as (x, y, z)
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append([lm.x, lm.y, lm.z])

                    # Flatten the landmarks for classification
                    flat_landmarks = np.array(landmarks).flatten()

                    # Predict the gesture
                    gesture = clf.predict([flat_landmarks])[0]
                    last_recognition_time = time.time()

                    # Display the gesture on the frame
                    cv2.putText(frame, f'Gesture: {gesture}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # Perform actions based on the gesture
                    if gesture == 'open':
                        print("Starting slideshow...")
                        pyautogui.hotkey('f5')  # Start slideshow
                    elif gesture == 'next':
                        print("Next slide...")
                        pyautogui.press('right')  # Next slide
                    elif gesture == 'stop':
                        print("Exiting slideshow...")
                        pyautogui.hotkey('esc')  # Exit slideshow

        # Show the frame
        cv2.imshow("Gesture-Controlled Slideshow", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
