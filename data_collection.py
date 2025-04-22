import cv2
import os
import time
import numpy as np
import mediapipe as mp

data_path = r"C:\Users\Karthik\Desktop\pro_testingn\dataset"
gestures = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
            "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
            "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3",
            "4", "5", "6", "7", "8", "9", "10"]
num_samples = 800
image_size = 300

class HandDataCollector:
    def __init__(self, data_path, gestures, num_samples, image_size=300):
        self.cap = cv2.VideoCapture(0)
        self.data_path = data_path
        self.gestures = gestures
        self.num_samples = num_samples
        self.image_size = image_size
        self.current_index = 0
        self.current_class = self.gestures[self.current_index]
        self.sample_count = 0
        self.is_recording = False

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.5)
        self.drawing = mp.solutions.drawing_utils

        os.makedirs(self.data_path, exist_ok=True)
        self.create_class_folder()

    def create_class_folder(self):
        self.class_dir = os.path.join(self.data_path, self.current_class)
        os.makedirs(self.class_dir, exist_ok=True)
        self.sample_count = len(os.listdir(self.class_dir))

    def capture_hand_image(self, frame, bbox):
        x, y, w, h = bbox
        offset = 20
        img_white = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 255

        x1 = max(x - offset, 0)
        y1 = max(y - offset, 0)
        x2 = min(x + w + offset, frame.shape[1])
        y2 = min(y + h + offset, frame.shape[0])

        img_crop = frame[y1:y2, x1:x2]
        aspect_ratio = h / w

        if aspect_ratio > 1:
            k = self.image_size / h
            w_cal = int(k * w)
            img_resize = cv2.resize(img_crop, (w_cal, self.image_size))
            w_gap = (self.image_size - w_cal) // 2
            img_white[:, w_gap:w_gap + w_cal] = img_resize
        else:
            k = self.image_size / w
            h_cal = int(k * h)
            img_resize = cv2.resize(img_crop, (self.image_size, h_cal))
            h_gap = (self.image_size - h_cal) // 2
            img_white[h_gap:h_gap + h_cal, :] = img_resize

        return img_white

    def switch_class(self):
        self.current_index = (self.current_index + 1) % len(self.gestures)
        self.current_class = self.gestures[self.current_index]
        self.create_class_folder()
        print(f"Switched to class '{self.current_class}'")

    def run(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                print("Failed to capture frame.")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(frame_rgb)

            h, w, _ = frame.shape
            img_white = None

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
                y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
                x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
                y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

                bbox = int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
                img_white = self.capture_hand_image(frame, bbox)

                self.drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                if self.is_recording and self.sample_count < self.num_samples:
                    save_path = os.path.join(self.class_dir, f"{self.sample_count}.jpg")
                    cv2.imwrite(save_path, img_white)
                    self.sample_count += 1
                elif self.sample_count >= self.num_samples:
                    print(f"Finished collecting {self.num_samples} images for '{self.current_class}'.")
                    self.is_recording = False

            info = f"Class: {self.current_class} | Index: {self.sample_count} | Recording: {'Yes' if self.is_recording else 'No'}"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Camera Feed", frame)
            if img_white is not None:
                cv2.imshow("Processed Hand", img_white)

            key = cv2.waitKey(1)

            if key == ord('r'):
                self.is_recording = not self.is_recording
                print("Recording toggled:", self.is_recording)
            elif key == ord('c'):
                self.switch_class()
            elif key == ord('q'):
                print("Exiting...")
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    collector = HandDataCollector(data_path=data_path, gestures=gestures, num_samples=num_samples)
    collector.run()
