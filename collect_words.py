import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import time
import csv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from train_model import extract_hand_features

#mp
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options, 
    running_mode=vision.RunningMode.VIDEO, 
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

WORD_TO_RECORD = input("Enter the word you want to record (e.g. HELLO): ").strip().upper()
SAMPLES_TO_COLLECT = 30  
FRAMES_PER_SAMPLE = 30  

cap = cv2.VideoCapture(0)
dataset = []
start_time_ms = int(time.time() * 1000)

print(f"\n--- Starting data collection for '{WORD_TO_RECORD}' ---")

for sample_idx in range(SAMPLES_TO_COLLECT):
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        
        cv2.putText(frame, f"Word: {WORD_TO_RECORD} | Sample {sample_idx + 1}/{SAMPLES_TO_COLLECT}", 
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Press 's' to start recording this sample", (30, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Data Collector", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    sequence_features = []
    
    #record 30 consecutive frames of hand features
    while len(sequence_features) < (FRAMES_PER_SAMPLE * 126):
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(time.time() * 1000) - start_time_ms
        
        res = detector.detect_for_video(mp_image, timestamp)
        
        if res.hand_landmarks:
            pts = np.array([[lm.x, lm.y, lm.z] for lm in res.hand_landmarks[0]])
            feats = extract_hand_features(pts)
        else:
            feats = [0.0] * 126  
            
        sequence_features.extend(feats)
        
        frames_recorded = len(sequence_features) // 126
        cv2.putText(frame, f"RECORDING... {frames_recorded}/{FRAMES_PER_SAMPLE}", 
                    (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Data Collector", frame)
        cv2.waitKey(30)

    dataset.append([WORD_TO_RECORD] + sequence_features)
    print(f"Sample {sample_idx + 1} recorded!")

cap.release()
cv2.destroyAllWindows()

#append data to asl_words.csv
if dataset:
    with open("asl_words.csv", mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(dataset)

print(f"\nSuccessfully saved {SAMPLES_TO_COLLECT} samples of '{WORD_TO_RECORD}' to asl_words.csv")