import cv2
import numpy as np
import os
import mediapipe as mp
import pickle

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define the folder for saving data
folder = "gesture_data"
if not os.path.exists(folder):
    os.makedirs(folder)  # Create folder if not exists

# Initialize hands model
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    # Initialize dictionary for storing gesture data
    gesture_data = {}

    print("Collecting gesture data. Press 'h' for 'Hi', 'l' for 'Hello', 's' for 'Stop'. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip the frame horizontally for better user experience
        frame = cv2.flip(frame, 1)

        # Convert the image to RGB (required by MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands model
        results = hands.process(rgb_frame)

        # Draw landmarks on the hands if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the landmarks' x, y, z coordinates
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])

                # Flatten the landmarks
                flat_landmarks = np.array(landmarks).flatten()

                # Show the frame
                cv2.imshow("Gesture Collection", frame)

                # Collect data when a key is pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('o'):  # 'h' for Hi
                    gesture_data.setdefault('open', []).append(flat_landmarks)
                    print("Recording 'open' gesture")
                elif key == ord('l'):  # 'l' for Hello
                    gesture_data.setdefault('hello', []).append(flat_landmarks)
                elif key == ord('n'):  # 'l' for Hello
                    gesture_data.setdefault('next', []).append(flat_landmarks)    
                    print("Recording 'next' gesture")
                elif key == ord('j'):  # 'l' for Hello
                    gesture_data.setdefault('jumping', []).append(flat_landmarks) 
                    print("Recording 'jumping' gesture")
                elif key == ord('s'):  # 's' for Stop
                    gesture_data.setdefault('stop', []).append(flat_landmarks)
                    print("Recording 'Stop' gesture")
                elif key == ord('q'):  # Press 'q' to quit
                    break

        # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # Save the gesture data dictionary
    data_path = os.path.join(os.getcwd(), folder, 'gesture_data.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(gesture_data, f)

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection complete. Data saved to gesture_data.pkl.")
