import cv2
import numpy as np
import mediapipe as mp
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import time
import csv
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from train_model import extract_hand_features
except ImportError:
    def extract_hand_features(landmarks):
        return [0.0] * 63

COLOR_BG_DARK = (30, 24, 32)
COLOR_CARD_BG = (45, 35, 50)
COLOR_PINK = (210, 160, 255)
COLOR_LAVENDER = (240, 190, 215)
COLOR_MINT = (200, 245, 180)
COLOR_CORAL = (140, 140, 255)
COLOR_YELLOW = (170, 235, 255)
COLOR_WHITE = (245, 245, 245)
COLOR_GRAY = (120, 110, 125)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

KEY_FACE_INDICES = [
    1,
    33, 133, 159, 145,
    362, 263, 386, 374,
    70, 63, 105, 66,
    300, 293, 334, 296,
    61, 291, 0, 17, 13, 14,
    78, 308, 82, 312
]

def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=15):
    x1, y1 = pt1
    x2, y2 = pt2
    w, h = x2 - x1, y2 - y1
    radius = min(radius, w // 2, h // 2)

    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

def draw_alpha_card(img, pt1, pt2, color, alpha=0.70, radius=15):
    overlay = img.copy()
    draw_rounded_rect(overlay, pt1, pt2, color, thickness=-1, radius=radius)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def extract_face_features(face_landmarks):
    if not face_landmarks:
        return [0.0] * (len(KEY_FACE_INDICES) * 3)
    
    nose_tip = np.array([face_landmarks[1].x, face_landmarks[1].y, face_landmarks[1].z])
    face_feats = []
    for idx in KEY_FACE_INDICES:
        lm = face_landmarks[idx]
        face_feats.extend([lm.x - nose_tip[0], lm.y - nose_tip[1], lm.z - nose_tip[2]])
    return face_feats

def is_valid_hand_shape(landmarks):
    wrist = np.array([landmarks[0].x, landmarks[0].y])
    middle_mcp = np.array([landmarks[9].x, landmarks[9].y])
    palm_size = np.linalg.norm(wrist - middle_mcp)
    return 0.02 < palm_size < 0.45

def append_sample_to_csv(filepath, row_data):
    try:
        with open(filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
    except Exception as e:
        backup_path = os.path.expanduser("~/asl_words_backup.csv")
        print(f"[warning] failed writing to {filepath}: {e}. backup saved to {backup_path}")
        with open(backup_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

def delete_last_csv_row(filepath):
    if not os.path.exists(filepath):
        return False
    try:
        with open(filepath, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return False
        with open(filepath, mode="w", encoding="utf-8") as f:
            f.writelines(lines[:-1])
        return True
    except Exception as e:
        print(f"[error] failed deleting last csv row: {e}")
        return False

def main():
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options, 
        running_mode=vision.RunningMode.VIDEO, 
        num_hands=2,
        min_hand_detection_confidence=0.50,
        min_hand_presence_confidence=0.50,
        min_tracking_confidence=0.50
    )
    detector = vision.HandLandmarker.create_from_options(options)

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.50,
        min_tracking_confidence=0.50
    )

    dummy_pts = np.zeros((21, 3))
    num_hand_features = len(extract_hand_features(dummy_pts))
    num_face_features = len(KEY_FACE_INDICES) * 3
    num_features_per_frame = num_hand_features + num_face_features

    frames_per_sample = 30
    target_total_features = frames_per_sample * num_features_per_frame

    csv_file = "asl_words.csv"
    if os.path.exists(csv_file):
        with open(csv_file, mode="r", encoding="utf-8") as f:
            existing_rows = sum(1 for _ in f)
        print(f"found existing '{csv_file}' with {existing_rows} rows.")
    else:
        print(f"'{csv_file}' not found. initializing new file on first save.")

    word_input = input("\nenter word to record (e.g. HELLO, SAD(e), or DONE/FINISH): ").strip()
    if not word_input:
        print("empty input. exiting.")
        return

    word_to_record = "/".join([w.strip().upper() for w in word_input.split("/") if w.strip()])

    try:
        samples_to_collect = int(input("enter number of samples to collect (default 30): ") or "30")
    except ValueError:
        samples_to_collect = 30

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    start_time_ms = int(time.time() * 1000)
    last_timestamp_ms = 0

    print(f"\n--- collecting word samples for '{word_to_record}' ---")
    print("controls: 's' = record | 'd' = delete last sample | 'q' = quit\n")

    sample_idx = 0
    status_msg = ""
    status_timer = 0

    while sample_idx < samples_to_collect:
        ready_to_record = False
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frame = cv2.flip(frame, 1)

            # header info card
            draw_alpha_card(frame, (20, 20), (480, 115), COLOR_CARD_BG, alpha=0.75, radius=18)
            draw_rounded_rect(frame, (20, 20), (480, 115), COLOR_PINK, thickness=2, radius=18)
            
            # target label badge
            draw_rounded_rect(frame, (35, 32), (150, 60), COLOR_PINK, thickness=-1, radius=10)
            cv2.putText(frame, "TARGET", (50, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BG_DARK, 2, cv2.LINE_AA)
            cv2.putText(frame, word_to_record, (165, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_WHITE, 2, cv2.LINE_AA)
            cv2.putText(frame, f"sample {sample_idx + 1} / {samples_to_collect}", (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_LAVENDER, 1, cv2.LINE_AA)

            # bottom controls card
            draw_alpha_card(frame, (20, 640), (580, 700), COLOR_CARD_BG, alpha=0.75, radius=15)
            cv2.putText(frame, "[S] Record", (40, 678), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_MINT, 2, cv2.LINE_AA)
            cv2.putText(frame, "|", (165, 678), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_GRAY, 1, cv2.LINE_AA)
            cv2.putText(frame, "[D] Delete Last", (185, 678), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_CORAL, 2, cv2.LINE_AA)
            cv2.putText(frame, "|", (355, 678), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_GRAY, 1, cv2.LINE_AA)
            cv2.putText(frame, "[Q] Quit", (375, 678), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_YELLOW, 2, cv2.LINE_AA)

            # alert notification toast
            if time.time() < status_timer:
                draw_alpha_card(frame, (20, 130), (500, 175), COLOR_CARD_BG, alpha=0.85, radius=12)
                draw_rounded_rect(frame, (20, 130), (500, 175), COLOR_CORAL, thickness=2, radius=12)
                cv2.putText(frame, f"~ {status_msg}", (35, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_YELLOW, 2, cv2.LINE_AA)

            cv2.imshow("word data collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                ready_to_record = True
                break
            elif key == ord('d'):
                if sample_idx > 0:
                    if delete_last_csv_row(csv_file):
                        sample_idx -= 1
                        status_msg = f"deleted sample {sample_idx + 1}"
                    else:
                        status_msg = "could not delete sample from csv"
                else:
                    status_msg = "no recorded samples in current session"
                status_timer = time.time() + 3.0
            elif key == ord('q'):
                print("\ncollection cancelled.")
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
            hand_feats = None
            if detection_result.hand_landmarks:
                valid_hands = [hand for hand in detection_result.hand_landmarks if is_valid_hand_shape(hand)]
                if valid_hands:
                    primary_hand = valid_hands[0]
                    pts = np.array([[lm.x, lm.y, lm.z] for lm in primary_hand])
                    hand_feats = extract_hand_features(pts)

                    for hand_landmarks in valid_hands:
                        for connection in HAND_CONNECTIONS:
                            start_p = (int(hand_landmarks[connection[0]].x * w), int(hand_landmarks[connection[0]].y * h))
                            end_p = (int(hand_landmarks[connection[1]].x * w), int(hand_landmarks[connection[1]].y * h))
                            cv2.line(frame, start_p, end_p, COLOR_LAVENDER, 2, cv2.LINE_AA)
                        for lm in hand_landmarks:
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (cx, cy), 5, COLOR_PINK, -1, cv2.LINE_AA)
                            cv2.circle(frame, (cx, cy), 2, COLOR_WHITE, -1, cv2.LINE_AA)

            if hand_feats is None:
                hand_feats = [0.0] * num_hand_features

            face_results = face_mesh.process(rgb_frame)
            face_lms = face_results.multi_face_landmarks[0].landmark if face_results.multi_face_landmarks else None
            face_feats = extract_face_features(face_lms)

            if face_lms:
                for idx in KEY_FACE_INDICES:
                    lm = face_lms[idx]
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, COLOR_MINT, -1, cv2.LINE_AA)

            frame_feats = list(hand_feats) + list(face_feats)
            sequence_features.extend(frame_feats)

            recorded_frames = len(sequence_features) // num_features_per_frame
            ratio = recorded_frames / frames_per_sample

            # recording card & rounded progress bar
            draw_alpha_card(frame, (20, 20), (520, 120), COLOR_CARD_BG, alpha=0.85, radius=18)
            draw_rounded_rect(frame, (20, 20), (520, 120), COLOR_CORAL, thickness=2, radius=18)
            
            cv2.circle(frame, (45, 52), 7, COLOR_CORAL, -1, cv2.LINE_AA)
            cv2.putText(frame, f"RECORDING '{word_to_record}'", (65, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 2, cv2.LINE_AA)
            cv2.putText(frame, f"{recorded_frames}/{frames_per_sample} frames", (380, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LAVENDER, 1, cv2.LINE_AA)

            bar_x1, bar_y1, bar_x2, bar_y2 = 40, 80, 500, 98
            draw_rounded_rect(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), COLOR_BG_DARK, thickness=-1, radius=9)
            fill_w = int((bar_x2 - bar_x1) * ratio)
            if fill_w > 18:
                draw_rounded_rect(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), COLOR_CORAL, thickness=-1, radius=9)

            cv2.imshow("word data collector", frame)
            cv2.waitKey(20)

        sample_row = [word_to_record] + sequence_features
        append_sample_to_csv(csv_file, sample_row)
        sample_idx += 1
        print(f"saved sample {sample_idx}/{samples_to_collect} for '{word_to_record}'")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nfinished. successfully added {sample_idx} samples to '{csv_file}'.")

if __name__ == "__main__":
    main()