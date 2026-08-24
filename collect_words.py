import cv2
import numpy as np
import mediapipe as mp
import time
import csv
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from train_model import extract_hand_features

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (5, 6), (6, 7), (7, 8),                 # index 
    (9, 10), (10, 11), (11, 12),            # middle 
    (13, 14), (14, 15), (15, 16),           # ring 
    (17, 18), (18, 19), (19, 20),           # pinky
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17) # palm
]

def is_valid_hand_shape(landmarks):
    # matches detector.py hand scale filter DONT TOUCH
    wrist = np.array([landmarks[0].x, landmarks[0].y])
    middle_mcp = np.array([landmarks[9].x, landmarks[9].y])
    palm_size = np.linalg.norm(wrist - middle_mcp)
    return 0.03 < palm_size < 0.40

def append_sample_to_csv(filepath, row_data):
    try:
        with open(filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
    except Exception as e:
        backup_path = os.path.expanduser("~/asl_words_backup.csv")
        print(f"[warning] could not write to {filepath} ({e}). writing to backup {backup_path}")
        with open(backup_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

def main():
    # setup mediapipe tasks hand landmarker matching detector.py parameters (2 hands)
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options, 
        running_mode=vision.RunningMode.VIDEO, 
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55
    )
    detector = vision.HandLandmarker.create_from_options(options)

    #126 features per frame * 30 frames = 3780
    dummy_pts = np.zeros((21, 3))
    num_features_per_frame = len(extract_hand_features(dummy_pts))
    frames_per_sample = 30
    target_total_features = frames_per_sample * num_features_per_frame

    csv_file = "asl_words.csv"
    if os.path.exists(csv_file):
        with open(csv_file, mode="r", encoding="utf-8") as f:
            existing_rows = sum(1 for _ in f)
        print(f"found existing '{csv_file}' with {existing_rows} total rows. new samples will append cleanly.")
    else:
        print(f"'{csv_file}' not found. a new file will be initialized on first record.")

    word_to_record = input("\nenter word to record (e.g. HELLO): ").strip().upper()
    if not word_to_record:
        print("empty input. exiting.")
        return

    try:
        samples_to_collect = int(input("enter number of samples to collect (default 30): ") or "30")
    except ValueError:
        samples_to_collect = 30

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    start_time_ms = int(time.time() * 1000)
    last_timestamp_ms = 0

    print(f"\n--- starting word data collection for '{word_to_record}' ---")
    print("controls: press 's' to record a sample, press 'q' to quit early.\n")

    sample_idx = 0
    while sample_idx < samples_to_collect:
        # prompt screen loop
        ready_to_record = False
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frame = cv2.flip(frame, 1)

            cv2.putText(frame, f"word: {word_to_record} | sample {sample_idx + 1}/{samples_to_collect}", 
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "press 's' to start sample recording | 'q' to exit", 
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            cv2.imshow("word data collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                ready_to_record = True
                break
            elif key == ord('q'):
                print("\ncollection canceled early by user.")
                cap.release()
                cv2.destroyAllWindows()
                return

        if not ready_to_record:
            break

        sequence_features = []
        while len(sequence_features) < target_total_features:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            frame_timestamp_ms = int(time.time() * 1000) - start_time_ms
            if frame_timestamp_ms <= last_timestamp_ms:
                frame_timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = frame_timestamp_ms

            detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

            feats = None
            if detection_result.hand_landmarks:
                valid_hands = [hand for hand in detection_result.hand_landmarks if is_valid_hand_shape(hand)]
                if valid_hands:
                    primary_hand = valid_hands[0]
                    pts = np.array([[lm.x, lm.y, lm.z] for lm in primary_hand])
                    feats = extract_hand_features(pts)

                    #skeleton overlays for both hands 
                    for hand_landmarks in valid_hands:
                        for connection in HAND_CONNECTIONS:
                            start_p = (int(hand_landmarks[connection[0]].x * w), int(hand_landmarks[connection[0]].y * h))
                            end_p = (int(hand_landmarks[connection[1]].x * w), int(hand_landmarks[connection[1]].y * h))
                            cv2.line(frame, start_p, end_p, (255, 255, 255), 2)
                        for lm in hand_landmarks:
                            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

            if feats is None:
                feats = [0.0] * num_features_per_frame

            sequence_features.extend(feats)

            recorded_frames = len(sequence_features) // num_features_per_frame
            cv2.putText(frame, f"RECORDING '{word_to_record}'... {recorded_frames}/{frames_per_sample}", 
                        (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            
            ratio = recorded_frames / frames_per_sample
            cv2.rectangle(frame, (30, 80), (330, 95), (100, 100, 100), 2)
            cv2.rectangle(frame, (30, 80), (30 + int(300 * ratio), 95), (0, 0, 255), -1)

            cv2.imshow("word data collector", frame)
            cv2.waitKey(20)

        # append sample row to csv cleanly
        sample_row = [word_to_record] + sequence_features
        append_sample_to_csv(csv_file, sample_row)
        sample_idx += 1
        print(f"saved sample {sample_idx}/{samples_to_collect} for '{word_to_record}'")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nfinished collection! successfully added {sample_idx} samples to '{csv_file}'.")

if __name__ == "__main__":
    main()