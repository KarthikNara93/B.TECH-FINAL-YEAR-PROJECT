import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

class DataCollector:
    def __init__(self, base_path):
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=1)
        
        self.offset = 20
        self.imgsize = 300
        
        self.base_path = base_path
        self.current_class = None
        self.class_samples = {}
        self.max_samples_per_class = 800
        
        self.is_recording = False
        
        os.makedirs(base_path, exist_ok=True)
        
        self.setup_classes()
        
    def setup_classes(self):
        self.classes = [
            'hello', 'thanks', 'yes', 'no', 
            'love', 'please', 'sorry', 'help'
        ]
        
        for cls in self.classes:
            class_path = os.path.join(self.base_path, cls)
            os.makedirs(class_path, exist_ok=True)
            self.class_samples[cls] = len(os.listdir(class_path))
    
    def process_hand(self, img):
        hands, img = self.detector.findHands(img)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            y1, y2 = max(y - self.offset, 0), min(y + h + self.offset, img.shape[0])
            x1, x2 = max(x - self.offset, 0), min(x + w + self.offset, img.shape[1])

            imgwhite = np.ones((self.imgsize, self.imgsize, 3), np.uint8) * 255
            imgcrop = img[y1:y2, x1:x2]

            if imgcrop.size == 0:
                return img, None

            aspectratio = h / w

            if aspectratio > 1:
                k = self.imgsize / h
                wcal = math.ceil(k * w)
                imgresize = cv2.resize(imgcrop, (wcal, self.imgsize))
                wgap = math.ceil((self.imgsize - wcal) / 2)
                imgwhite[:, wgap:wcal + wgap] = imgresize
            else:
                k = self.imgsize / w
                hcal = math.ceil(k * h)
                imgresize = cv2.resize(imgcrop, (self.imgsize, hcal))
                hgap = math.ceil((self.imgsize - hcal) / 2)
                imgwhite[hgap:hgap + hcal, :] = imgresize

            return img, imgwhite
        
        return img, None
    
    def save_sample(self, imgwhite):
        if not self.current_class:
            print("No class selected. Press a class key first.")
            return False
        
        if self.class_samples[self.current_class] >= self.max_samples_per_class:
            print(f"Max samples reached for {self.current_class}")
            return False
        
        filename = f'{time.time()}.jpg'
        filepath = os.path.join(self.base_path, self.current_class, filename)
        cv2.imwrite(filepath, imgwhite)
        
        self.class_samples[self.current_class] += 1
        
        print(f"Saved {filename} for class {self.current_class}")
        print(f"Current samples for {self.current_class}: {self.class_samples[self.current_class]}")
        
        return True
    
    def run(self):
        while True:
            success, img = self.cap.read()
            if not success:
                continue
            
            img, imgwhite = self.process_hand(img)
            
            if self.current_class:
                cv2.putText(img, 
                    f"Class: {self.current_class} (Samples: {self.class_samples[self.current_class]}/800)", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
            
            cv2.imshow("Main Camera", img)
            if imgwhite is not None:
                cv2.imshow("Hand Crop", imgwhite)
            
            key = cv2.waitKey(1)
            
            if key == 27: 
                break
            
            elif key == ord('r'):
                self.is_recording = not self.is_recording
                print("Recording", "Started" if self.is_recording else "Stopped")
            
            elif key == ord('1'):
                self.current_class = self.classes[0]
                print(f"Selected class: {self.current_class}")
            
            elif key == ord('2'):
                self.current_class = self.classes[1]
                print(f"Selected class: {self.current_class}")
            
            
            elif key == ord('s') and self.is_recording and imgwhite is not None:
                self.save_sample(imgwhite)
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    base_path = r"C:\Users\Karthik\Desktop\signlanguagedetection\data"
    collector = DataCollector(base_path)
    collector.run()