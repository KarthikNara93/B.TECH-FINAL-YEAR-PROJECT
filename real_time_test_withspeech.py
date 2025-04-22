import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyttsx3
from tensorflow.keras.models import load_model

# Load trained model
model = load_model(r"c:\Users\Karthik\Desktop\project_testing_2\cnn-lstm-model.h5")

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Define gesture labels
gesture_labels= ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
            "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
            "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3",
            "4", "5", "6", "7", "8", "9", "10"]  # Define gesture labels

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start webcam
cap = cv2.VideoCapture(0)
last_spoken = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for natural interaction
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract hand landmarks
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            landmarks = landmarks.reshape(1, 1, -1)  # Reshape for LSTM input

            # Predict gesture
            prediction = model.predict(landmarks)
            predicted_label = np.argmax(prediction)
            gesture = gesture_labels[predicted_label]

            # Draw landmarks on hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Display predicted gesture
            cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Speak the gesture (only once per change to avoid repetition)
            if gesture != last_spoken:
                speak(gesture)
                last_spoken = gesture

    # Show frame
    cv2.imshow("Real-Time Sign Recognition", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Real-time gesture recognition stopped.")