import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import os
import joblib
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from train_model import extract_hand_features
except ImportError:
    def extract_hand_features(landmarks):
        return [0.0] * 63  

# soft color palette for the opencv ui
COLOR_BG_CARD = (245, 245, 245)
COLOR_BORDER = (210, 210, 210)
COLOR_TEXT_DARK = (40, 40, 40)
COLOR_TEXT_MUTED = (120, 120, 120)
COLOR_SAGE = (120, 170, 130)
COLOR_ROSE = (120, 120, 220)
COLOR_TERRACOTTA = (70, 110, 210)
COLOR_SAND = (180, 210, 230)
COLOR_OVERLAY_BG = (250, 250, 250)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (5, 6), (6, 7), (7, 8),                 # index finger
    (9, 10), (10, 11), (11, 12),            # middle finger
    (13, 14), (14, 15), (15, 16),           # ring finger
    (17, 18), (18, 19), (19, 20),           # pinky finger
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17) # palm base
]

# selected facial landmark indices for expression intensity tracking
KEY_FACE_INDICES = [
    1,                  # nose tip anchor point
    33, 133, 159, 145,  # left eye bounds
    362, 263, 386, 374, # right eye bounds
    70, 63, 105, 66,    # left eyebrow
    300, 293, 334, 296, # right eyebrow
    61, 291, 0, 17, 13, 14,
    78, 308, 82, 312    # lip curvature
]

def extract_face_features(face_landmarks):
    if not face_landmarks:
        return [0.0] * (len(KEY_FACE_INDICES) * 3)
    nose_tip = np.array([face_landmarks[1].x, face_landmarks[1].y, face_landmarks[1].z])
    face_feats = []
    for idx in KEY_FACE_INDICES:
        lm = face_landmarks[idx]
        face_feats.extend([lm.x - nose_tip[0], lm.y - nose_tip[1], lm.z - nose_tip[2]])
    return face_feats

def calculate_facial_intensity(face_landmarks):
    if not face_landmarks:
        return 1.0, "NEUTRAL"
    
    def get_pt(idx):
        lm = face_landmarks[idx]
        return np.array([lm.x, lm.y])

    # normalize distance using outer eye span as base face scale
    left_eye_outer = get_pt(33)
    right_eye_outer = get_pt(263)
    face_scale = np.linalg.norm(left_eye_outer - right_eye_outer)
    if face_scale == 0:
        return 1.0, "NEUTRAL"

    # distance measurements normalized by face scale
    upper_lip = get_pt(13)
    lower_lip = get_pt(14)
    mouth_height = np.linalg.norm(upper_lip - lower_lip) / face_scale

    left_eyebrow = get_pt(105)
    left_eye = get_pt(159)
    eyebrow_dist = np.linalg.norm(left_eyebrow - left_eye) / face_scale

    left_eye_top = get_pt(159)
    left_eye_bottom = get_pt(145)
    eye_openness = np.linalg.norm(left_eye_top - left_eye_bottom) / face_scale

    neutral_mouth = 0.08
    neutral_eyebrow = 0.22
    neutral_eye = 0.12

    # calculate relative displacement deltas
    mouth_delta = max(0.0, mouth_height - neutral_mouth)
    eyebrow_delta = abs(eyebrow_dist - neutral_eyebrow)
    eye_delta = max(0.0, eye_openness - neutral_eye)

    #weight deltas to scores
    raw_score = (mouth_delta * 2.5) + (eyebrow_delta * 3.0) + (eye_delta * 2.0)
    intensity_multiplier = 1.0 + min(max(raw_score, 0.0), 1.5)

    if intensity_multiplier < 1.25:
        level = "NEUTRAL"
    elif intensity_multiplier < 1.60:
        level = "MODERATE"
    else:
        level = "EXTREME"

    return round(intensity_multiplier, 2), level

def is_valid_hand_shape(landmarks):
    wrist = np.array([landmarks[0].x, landmarks[0].y])
    middle_mcp = np.array([landmarks[9].x, landmarks[9].y])
    palm_size = np.linalg.norm(wrist - middle_mcp)
    return 0.02 < palm_size < 0.45

def is_open_hand(hand_landmarks):
    wrist = hand_landmarks[0]
    tips = [4, 8, 12, 16, 20]
    mcps = [2, 5, 9, 13, 17]
    open_fingers = 0
    for tip, mcp in zip(tips, mcps):
        dist_tip = np.linalg.norm([hand_landmarks[tip].x - wrist.x, hand_landmarks[tip].y - wrist.y])
        dist_mcp = np.linalg.norm([hand_landmarks[mcp].x - wrist.x, hand_landmarks[mcp].y - wrist.y])
        if dist_tip > dist_mcp:
            open_fingers += 1
    return open_fingers >= 4

def is_two_open_hands(hand1, hand2):
    return is_open_hand(hand1) and is_open_hand(hand2)

def aggregate_sequence(sequence_matrix):
    seq = np.array(sequence_matrix, dtype=np.float32)
    mean_f = np.mean(seq, axis=0)
    std_f = np.std(seq, axis=0)
    delta_f = seq[-1] - seq[0]
    max_f = np.max(seq, axis=0)
    min_f = np.min(seq, axis=0)
    return np.hstack([mean_f, std_f, delta_f, max_f, min_f])

def transform_tense(sentence, tense_target):
    # simple text formatting hook for tenses
    clean_sent = sentence.strip()
    if not clean_sent:
        return ""
    if tense_target == "NOW":
        return f"{clean_sent[:-1] if clean_sent[-1] in '.!?' else clean_sent} right now."
    elif tense_target == "PAST":
        return f"Previously, {clean_sent.lower()}"
    elif tense_target == "FUTURE":
        return f"Will {clean_sent.lower()}"
    return clean_sent

def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=10):
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1
    r = min(radius, abs(w) // 2, abs(h) // 2)

    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)

def draw_pill_button(img, pt1, pt2, bg_color, text, text_color=(40, 40, 40), font_scale=0.5):
    draw_rounded_rect(img, pt1, pt2, bg_color, thickness=-1, radius=12)
    draw_rounded_rect(img, pt1, pt2, COLOR_BORDER, thickness=1, radius=12)
    t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
    cx = (pt1[0] + pt2[0]) // 2
    cy = (pt1[1] + pt2[1]) // 2
    cv2.putText(img, text, (cx - t_size[0] // 2, cy + t_size[1] // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)

def draw_progress_bar(img, pt1, pt2, ratio, color=COLOR_TERRACOTTA):
    draw_rounded_rect(img, pt1, pt2, (220, 220, 220), thickness=-1, radius=4)
    w = pt2[0] - pt1[0]
    fill_x2 = pt1[0] + int(w * min(max(ratio, 0.0), 1.0))
    if fill_x2 > pt1[0] + 4:
        draw_rounded_rect(img, pt1, (fill_x2, pt2[1]), color, thickness=-1, radius=4)

# tts & background inference workers
class TextToSpeechEngine:
    def __init__(self):
        self.voices = ["Default Neutral", "Expressive"]
        self.voice_idx = 0
        self.current_voice_label = self.voices[self.voice_idx]

    def speak(self, text, tone="neutral", intensity=1.0):
        # logging tts speech output along with intensity context
        intensity_tag = f" [{intensity}x]" if intensity > 1.2 else ""
        print(f"[TTS ({tone}){intensity_tag}]: {text}")

    def toggle_voice(self):
        self.voice_idx = (self.voice_idx + 1) % len(self.voices)
        self.current_voice_label = self.voices[self.voice_idx]

class BackgroundAIThread:
    def __init__(self, letter_model=None):
        self.letter_model = letter_model
        self.predicted_letter = "-"
        self.result_lock = threading.Lock()
        self.running = True

    def update_data(self, features, hand_x=None):
        if features is not None and self.letter_model is not None:
            try:
                pred = self.letter_model.predict([features])[0]
                with self.result_lock:
                    self.predicted_letter = str(pred).upper()
            except Exception:
                with self.result_lock:
                    self.predicted_letter = "-"
        else:
            with self.result_lock:
                self.predicted_letter = "-"

mouse_click_pos = None
def on_mouse_click(event, x, y, flags, param):
    global mouse_click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_click_pos = (x, y)

def main():
    global mouse_click_pos

    # load trained model files
    asl_word_model = None
    if os.path.exists("asl_word_model.pkl"):
        asl_word_model = joblib.load("asl_word_model.pkl")
        print("loaded asl_word_model.pkl")

    letter_model = None
    if os.path.exists("asl_letter_model.pkl"):
        letter_model = joblib.load("asl_letter_model.pkl")
    elif os.path.exists("model.pkl"):
        letter_model = joblib.load("model.pkl")

    bg_ai = BackgroundAIThread(letter_model=letter_model)
    tts = TextToSpeechEngine()

    # setup mediapipe tasks
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

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.50,
        min_tracking_confidence=0.50
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "ASL Gesture Translation System"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_click)

    # application states
    current_mode = "SPELL"
    word_sequence_buffer = deque(maxlen=30)
    last_word_pred_time = 0.0
    WORD_COOLDOWN_DURATION = 3.0

    is_recording = False
    selecting_punctuation = False
    punct_sub = None
    selecting_tense = False
    is_converting_tense = False
    temp_sentence = ""

    selecting_synonym = False
    synonym_options = []

    selected_tone = "neutral"

    current_word = ""
    finished_word = ""
    word_history = []

    open_hand_start_time = None
    open_hand_triggered = False
    space_start_time = None
    space_triggered = False

    letter_hold_start_time = None
    current_holding_letter = None
    prev_wrist_pos = None

    HOLD_LETTER_DURATION = 2.0 
    TOGGLE_GESTURE_DURATION = 1.2 
    ACTION_GESTURE_DURATION = 1.0

    start_time_ms = int(time.time() * 1000)
    last_timestamp_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2

        with bg_ai.result_lock:
            predicted_letter = bg_ai.predicted_letter

        # handle mouse interactions across interface overlays
        if mouse_click_pos is not None:
            mx, my = mouse_click_pos
            mouse_click_pos = None

            if (w - 550) <= mx <= (w - 420) and 20 <= my <= 55:
                is_recording = not is_recording
                if is_recording:
                    current_word = ""
                    finished_word = ""
                    selecting_punctuation = False
                    punct_sub = None
                    selecting_tense = False
                    selecting_synonym = False
                    word_sequence_buffer.clear()
                    last_word_pred_time = 0.0
                else:
                    if current_word.strip():
                        selecting_punctuation = True
                        punct_sub = None
                    else:
                        finished_word = ""
                open_hand_start_time = None

            elif (w - 410) <= mx <= (w - 280) and 20 <= my <= 55:
                current_mode = "WORD" if current_mode == "SPELL" else "SPELL"
                selecting_synonym = False
                word_sequence_buffer.clear()
                last_word_pred_time = 0.0

            elif (w - 140) <= mx <= (w - 20) and 20 <= my <= 55:
                current_word = ""
                finished_word = ""
                selecting_punctuation = False
                punct_sub = None
                selecting_tense = False
                selecting_synonym = False

            elif (w - 270) <= mx <= (w - 150) and 20 <= my <= 55:
                current_word = current_word[:-1]

            elif selecting_synonym:
                btn_y1, btn_y2 = cy - 10, cy + 50
                if (cx - 210) <= mx <= (cx - 10) and btn_y1 <= my <= btn_y2 and len(synonym_options) >= 1:
                    chosen = synonym_options[0]
                    current_word += f"{chosen} "
                    tts.speak(chosen, tone=selected_tone)
                    selecting_synonym = False
                    synonym_options = []
                elif (cx + 10) <= mx <= (cx + 210) and btn_y1 <= my <= btn_y2 and len(synonym_options) >= 2:
                    chosen = synonym_options[1]
                    current_word += f"{chosen} "
                    tts.speak(chosen, tone=selected_tone)
                    selecting_synonym = False
                    synonym_options = []

            elif selecting_punctuation:
                btn_y1, btn_y2 = cy - 10, cy + 80
                if punct_sub is None:
                    if (cx - 320) <= mx <= (cx - 120) and btn_y1 <= my <= btn_y2:
                        punct_sub = "PERIOD"
                    elif (cx - 100) <= mx <= (cx + 100) and btn_y1 <= my <= btn_y2:
                        punct_sub = "EXCLAMATION"
                    elif (cx + 120) <= mx <= (cx + 320) and btn_y1 <= my <= btn_y2:
                        punct_sub = "QUESTION"

                elif punct_sub == "PERIOD":
                    if (cx - 300) <= mx <= (cx - 110) and btn_y1 <= my <= btn_y2:
                        temp_sentence, selected_tone = current_word.strip() + ".", "neutral"
                        selecting_punctuation, punct_sub, selecting_tense = False, None, True
                    elif (cx - 95) <= mx <= (cx + 95) and btn_y1 <= my <= btn_y2:
                        temp_sentence, selected_tone = current_word.strip() + ".", "sad"
                        selecting_punctuation, punct_sub, selecting_tense = False, None, True
                    elif (cx + 110) <= mx <= (cx + 300) and btn_y1 <= my <= btn_y2:
                        temp_sentence, selected_tone = current_word.strip() + ".", "sarcastic"
                        selecting_punctuation, punct_sub, selecting_tense = False, None, True

                elif punct_sub == "EXCLAMATION":
                    if (cx - 200) <= mx <= (cx - 10) and btn_y1 <= my <= btn_y2:
                        temp_sentence, selected_tone = current_word.strip() + "!", "happy"
                        selecting_punctuation, punct_sub, selecting_tense = False, None, True
                    elif (cx + 10) <= mx <= (cx + 200) and btn_y1 <= my <= btn_y2:
                        temp_sentence, selected_tone = current_word.strip() + "!", "angry"
                        selecting_punctuation, punct_sub, selecting_tense = False, None, True

                elif punct_sub == "QUESTION":
                    if (cx - 200) <= mx <= (cx - 10) and btn_y1 <= my <= btn_y2:
                        temp_sentence, selected_tone = current_word.strip() + "?", "surprised"
                        selecting_punctuation, punct_sub, selecting_tense = False, None, True
                    elif (cx + 10) <= mx <= (cx + 200) and btn_y1 <= my <= btn_y2:
                        temp_sentence, selected_tone = current_word.strip() + "?", "sarcastic"
                        selecting_punctuation, punct_sub, selecting_tense = False, None, True

            elif selecting_tense and not is_converting_tense:
                chosen_tense = None
                if (cy - 10) <= my <= (cy + 35):
                    if (cx - 210) <= mx <= (cx - 10): chosen_tense = "NOW"
                    elif (cx + 10) <= mx <= (cx + 210): chosen_tense = "PAST"
                elif (cy + 50) <= my <= (cy + 95):
                    if (cx - 210) <= mx <= (cx - 10): chosen_tense = "FUTURE"
                    elif (cx + 10) <= mx <= (cx + 210): chosen_tense = "ORIGINAL"

                if chosen_tense:
                    selecting_tense = False
                    is_converting_tense = True

                    def process_tense_and_speak(t_tense, raw_sent, tone):
                        nonlocal finished_word, is_converting_tense
                        final_sent = transform_tense(raw_sent, t_tense)
                        finished_word = final_sent
                        if final_sent:
                            word_history.append(final_sent)
                            tts.speak(final_sent, tone)
                        is_converting_tense = False

                    threading.Thread(
                        target=process_tense_and_speak, 
                        args=(chosen_tense, temp_sentence, selected_tone), 
                        daemon=True
                    ).start()

            elif not is_recording and finished_word:
                btn_y1, btn_y2 = cy + 20, cy + 65
                if (cx - 305) <= mx <= (cx - 115) and btn_y1 <= my <= btn_y2:
                    tts.speak(finished_word, selected_tone)
                elif (cx - 95) <= mx <= (cx + 95) and btn_y1 <= my <= btn_y2:
                    if word_history and word_history[-1] == finished_word: word_history.pop()
                    finished_word = ""
                    selecting_punctuation, punct_sub = True, None
                elif (cx + 115) <= mx <= (cx + 305) and btn_y1 <= my <= btn_y2:
                    finished_word = ""
                    current_word = ""

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        frame_timestamp_ms = int(time.time() * 1000) - start_time_ms
        if frame_timestamp_ms <= last_timestamp_ms:
            frame_timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = frame_timestamp_ms

        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
        face_results = face_mesh.process(rgb_frame)
        face_lms = face_results.multi_face_landmarks[0].landmark if face_results.multi_face_landmarks else None
        face_feats = extract_face_features(face_lms)
        
        # calculate facial intensity multiplier from facial landmarks
        intensity_mult, intensity_label = calculate_facial_intensity(face_lms)

        hand_detected = False
        space_gesture_detected = False
        open_hand = False
        features = None

        if detection_result.hand_landmarks:
            valid_hands = [h for h in detection_result.hand_landmarks if is_valid_hand_shape(h)]
            if valid_hands:
                hand_detected = True
                primary_hand = valid_hands[0]
                curr_wrist = np.array([primary_hand[0].x, primary_hand[0].y])
                hand_movement = np.linalg.norm(curr_wrist - prev_wrist_pos) if prev_wrist_pos is not None else 0.0
                prev_wrist_pos = curr_wrist

                if len(valid_hands) >= 2:
                    space_gesture_detected = is_two_open_hands(valid_hands[0], valid_hands[1])

                if not space_gesture_detected:
                    open_hand = is_open_hand(primary_hand)

                if is_recording and space_gesture_detected:
                    if space_start_time is None: space_start_time = time.time()
                    elif (time.time() - space_start_time >= ACTION_GESTURE_DURATION) and not space_triggered:
                        current_word += " "
                        space_triggered = True
                else:
                    space_start_time, space_triggered = None, False

                if current_mode == "SPELL" and open_hand and not space_gesture_detected and hand_movement < 0.035:
                    if open_hand_start_time is None: open_hand_start_time = time.time()
                    elif (time.time() - open_hand_start_time >= TOGGLE_GESTURE_DURATION) and not open_hand_triggered:
                        open_hand_triggered = True
                        is_recording = not is_recording
                        if is_recording:
                            current_word, finished_word = "", ""
                            selecting_punctuation, punct_sub, selecting_tense, selecting_synonym = False, None, False, False
                            word_sequence_buffer.clear()
                            last_word_pred_time = 0.0
                        else:
                            if current_word.strip():
                                selecting_punctuation, punct_sub = True, None
                            else: finished_word = ""
                        open_hand_start_time = None
                else:
                    open_hand_start_time, open_hand_triggered = None, False

                pts = np.array([[lm.x, lm.y, lm.z] for lm in primary_hand])
                features = extract_hand_features(pts)
                bg_ai.update_data(features, primary_hand[0].x)

                for hand_landmarks in valid_hands:
                    for connection in HAND_CONNECTIONS:
                        start_p = (int(hand_landmarks[connection[0]].x * w), int(hand_landmarks[connection[1]].y * h))
                        end_p = (int(hand_landmarks[connection[1]].x * w), int(hand_landmarks[connection[1]].y * h))
                        cv2.line(frame, start_p, end_p, (230, 235, 240), 2, cv2.LINE_AA)
                    for landmark in hand_landmarks:
                        cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 4, COLOR_TERRACOTTA, -1, cv2.LINE_AA)
            else:
                bg_ai.update_data(None, None)
                open_hand_start_time, space_start_time, letter_hold_start_time, current_holding_letter, prev_wrist_pos = None, None, None, None, None
        else:
            bg_ai.update_data(None, None)
            open_hand_start_time, space_start_time, letter_hold_start_time, current_holding_letter, prev_wrist_pos = None, None, None, None, None

        if is_recording and not open_hand and not space_gesture_detected and hand_detected and not selecting_synonym:
            if current_mode == "SPELL":
                if predicted_letter == current_holding_letter:
                    if time.time() - letter_hold_start_time >= HOLD_LETTER_DURATION:
                        current_word += predicted_letter
                        tts.speak(predicted_letter, tone="neutral", intensity=intensity_mult)
                        letter_hold_start_time = time.time()  
                else:
                    current_holding_letter = predicted_letter
                    letter_hold_start_time = time.time()

            elif current_mode == "WORD" and asl_word_model is not None and features is not None:
                if time.time() - last_word_pred_time >= WORD_COOLDOWN_DURATION:
                    combined_frame_feats = list(features) + list(face_feats)
                    word_sequence_buffer.append(combined_frame_feats)

                    if len(word_sequence_buffer) == 30:
                        aggregated_vec = aggregate_sequence(word_sequence_buffer)
                        try:
                            predicted_word = asl_word_model.predict([aggregated_vec])[0]
                            if predicted_word:
                                # adjust intensity prefix if facial intensity is high
                                word_str = predicted_word.strip()
                                if intensity_label == "EXTREME":
                                    word_str = f"VERY {word_str}"

                                if "/" in word_str:
                                    synonym_options = [w.strip() for w in word_str.split("/") if w.strip()]
                                    selecting_synonym = True
                                else:
                                    current_word += f"{word_str} "
                                    tts.speak(word_str, tone=selected_tone, intensity=intensity_mult)
                        except Exception as e:
                            print("word prediction error:", e)

                        word_sequence_buffer.clear()
                        last_word_pred_time = time.time()
                else:
                    word_sequence_buffer.clear()
        else:
            letter_hold_start_time, current_holding_letter = None, None

        # render action buttons
        rec_bg = COLOR_SAGE if not is_recording else COLOR_ROSE
        rec_txt_color = (255, 255, 255)
        rec_btn_text = "STOP" if is_recording else "START"
        draw_pill_button(frame, (w - 550, 20), (w - 420, 55), rec_bg, rec_btn_text, text_color=rec_txt_color, font_scale=0.55)

        mode_bg = COLOR_TERRACOTTA if current_mode == "SPELL" else COLOR_SAND
        mode_txt_color = (255, 255, 255) if current_mode == "SPELL" else COLOR_TEXT_DARK
        draw_pill_button(frame, (w - 410, 20), (w - 280, 55), mode_bg, f"MODE: {current_mode}", text_color=mode_txt_color, font_scale=0.45)

        draw_pill_button(frame, (w - 270, 20), (w - 150, 55), COLOR_SAND, "DELETE", text_color=COLOR_TEXT_DARK, font_scale=0.55)
        draw_pill_button(frame, (w - 140, 20), (w - 20, 55), COLOR_ROSE, "CLEAR", text_color=(255, 255, 255), font_scale=0.55)

        # instructions panel
        box_x1, box_y1 = w - 300, 75
        box_x2, box_y2 = w - 20, 310
        draw_rounded_rect(frame, (box_x1, box_y1), (box_x2, box_y2), COLOR_BG_CARD, thickness=-1, radius=12)
        draw_rounded_rect(frame, (box_x1, box_y1), (box_x2, box_y2), COLOR_BORDER, thickness=1, radius=12)

        instructions = [
            "CONTROLS",
            "Start/Stop: Click START/STOP",
            "Toggle Mode: Click MODE / 'm'",
            "Start Word Mode: Press 's'",
            "Palm Start: Hold open palm",
            "Space: Hold 2 open palms",
            "Delete: Click DELETE",
            "Clear All: Click CLEAR",
            "Type Letter: Hold sign 2s",
            "Toggle Voice: Press 'v'",
            "Quit App: Press 'q'"
        ]

        for idx, line_text in enumerate(instructions):
            y_pos = box_y1 + 22 + (idx * 20)
            if idx == 0:
                cv2.putText(frame, line_text, (box_x1 + 14, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_DARK, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, line_text, (box_x1 + 14, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        # status cards
        draw_rounded_rect(frame, (20, 20), (220, 65), COLOR_BG_CARD, thickness=-1, radius=10)
        draw_rounded_rect(frame, (20, 20), (220, 65), COLOR_BORDER, thickness=1, radius=10)
        cv2.putText(frame, "CURRENT SIGN", (32, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{predicted_letter}", (32, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT_DARK, 2, cv2.LINE_AA)

        draw_rounded_rect(frame, (230, 20), (430, 65), COLOR_BG_CARD, thickness=-1, radius=10)
        draw_rounded_rect(frame, (230, 20), (430, 65), COLOR_BORDER, thickness=1, radius=10)
        cv2.putText(frame, "VOICE ENGINE", (242, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{tts.current_voice_label}", (242, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TERRACOTTA, 2, cv2.LINE_AA)

        # facial expression intensity card
        draw_rounded_rect(frame, (440, 20), (640, 65), COLOR_BG_CARD, thickness=-1, radius=10)
        draw_rounded_rect(frame, (440, 20), (640, 65), COLOR_BORDER, thickness=1, radius=10)
        cv2.putText(frame, "EMOTION INTENSITY", (452, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{intensity_label} ({intensity_mult}x)", (452, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_SAGE if intensity_label == "NEUTRAL" else COLOR_TERRACOTTA, 2, cv2.LINE_AA)

        if is_recording:
            draw_rounded_rect(frame, (20, 80), (430, 160), COLOR_BG_CARD, thickness=-1, radius=12)
            draw_rounded_rect(frame, (20, 80), (430, 160), COLOR_BORDER, thickness=1, radius=12)
            cv2.putText(frame, f"RECORDING ({current_mode})", (32, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_SAGE, 2, cv2.LINE_AA)
            disp_word = f"{current_word}_" if current_word else "..."
            cv2.putText(frame, disp_word, (32, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT_DARK, 2, cv2.LINE_AA)

        if is_recording and not open_hand and not space_gesture_detected and hand_detected and not selecting_synonym:
            if current_mode == "SPELL" and current_holding_letter and current_holding_letter != "-" and letter_hold_start_time:
                letter_elapsed = min(time.time() - letter_hold_start_time, HOLD_LETTER_DURATION)
                l_ratio = letter_elapsed / HOLD_LETTER_DURATION
                cv2.putText(frame, f"Holding '{current_holding_letter}'...", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_DARK, 1, cv2.LINE_AA)
                draw_progress_bar(frame, (20, 188), (220, 200), l_ratio, color=COLOR_TERRACOTTA)

            elif current_mode == "WORD" and asl_word_model is not None:
                buf_len = len(word_sequence_buffer)
                w_ratio = buf_len / 30.0
                cv2.putText(frame, f"Capturing gesture ({buf_len}/30)...", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_DARK, 1, cv2.LINE_AA)
                draw_progress_bar(frame, (20, 188), (220, 200), w_ratio, color=COLOR_SAGE)

        # word intent / synonym selection modal
        if selecting_synonym:
            box_w, box_h = 520, 160
            m_x1, m_y1 = cx - box_w // 2, cy - box_h // 2
            m_x2, m_y2 = cx + box_w // 2, cy + box_h // 2
            draw_rounded_rect(frame, (m_x1, m_y1), (m_x2, m_y2), COLOR_OVERLAY_BG, thickness=-1, radius=16)
            draw_rounded_rect(frame, (m_x1, m_y1), (m_x2, m_y2), COLOR_BORDER, thickness=1, radius=16)

            title = "SELECT WORD INTENT"
            t_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
            cv2.putText(frame, title, (cx - t_size[0] // 2, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT_DARK, 2, cv2.LINE_AA)

            btn_h = 45
            if len(synonym_options) >= 1:
                opt1 = synonym_options[0]
                draw_pill_button(frame, (cx - 210, cy - 10), (cx - 10, cy - 10 + btn_h), COLOR_SAGE, opt1, text_color=(255, 255, 255), font_scale=0.6)

            if len(synonym_options) >= 2:
                opt2 = synonym_options[1]
                draw_pill_button(frame, (cx + 10, cy - 10), (cx + 210, cy - 10 + btn_h), COLOR_TERRACOTTA, opt2, text_color=(255, 255, 255), font_scale=0.6)

        # punctuation selection modal
        if selecting_punctuation:
            box_w, box_h = 700, 190
            m_x1, m_y1 = cx - box_w // 2, cy - box_h // 2
            m_x2, m_y2 = cx + box_w // 2, cy + box_h // 2
            draw_rounded_rect(frame, (m_x1, m_y1), (m_x2, m_y2), COLOR_OVERLAY_BG, thickness=-1, radius=16)
            draw_rounded_rect(frame, (m_x1, m_y1), (m_x2, m_y2), COLOR_BORDER, thickness=1, radius=16)

            btn_y1 = cy - 10
            btn_h = 80

            if punct_sub is None:
                title = "SELECT SENTENCE PUNCTUATION"
                t_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
                cv2.putText(frame, title, (cx - t_size[0] // 2, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT_DARK, 2, cv2.LINE_AA)

                draw_pill_button(frame, (cx - 320, btn_y1), (cx - 120, btn_y1 + btn_h), COLOR_SAND, ". (PERIOD)", font_scale=0.55)
                draw_pill_button(frame, (cx - 100, btn_y1), (cx + 100, btn_y1 + btn_h), COLOR_TERRACOTTA, "! (EXCLAMATION)", text_color=(255, 255, 255), font_scale=0.5)
                draw_pill_button(frame, (cx + 120, btn_y1), (cx + 320, btn_y1 + btn_h), COLOR_SAND, "? (QUESTION)", font_scale=0.55)

            elif punct_sub == "PERIOD":
                title = "SELECT EXPRESSION TONE"
                t_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
                cv2.putText(frame, title, (cx - t_size[0] // 2, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT_DARK, 2, cv2.LINE_AA)

                draw_pill_button(frame, (cx - 300, btn_y1), (cx - 110, btn_y1 + btn_h), COLOR_SAND, "NEUTRAL", font_scale=0.55)
                draw_pill_button(frame, (cx - 95, btn_y1), (cx + 95, btn_y1 + btn_h), COLOR_SAND, "SAD", font_scale=0.55)
                draw_pill_button(frame, (cx + 110, btn_y1), (cx + 300, btn_y1 + btn_h), COLOR_TERRACOTTA, "SARCASTIC", text_color=(255, 255, 255), font_scale=0.55)

            elif punct_sub == "EXCLAMATION":
                title = "SELECT EXPRESSION TONE"
                t_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
                cv2.putText(frame, title, (cx - t_size[0] // 2, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT_DARK, 2, cv2.LINE_AA)

                draw_pill_button(frame, (cx - 200, btn_y1), (cx - 10, btn_y1 + btn_h), COLOR_SAGE, "HAPPY", text_color=(255, 255, 255), font_scale=0.6)
                draw_pill_button(frame, (cx + 10, btn_y1), (cx + 200, btn_y1 + btn_h), COLOR_ROSE, "ANGRY", text_color=(255, 255, 255), font_scale=0.6)

            elif punct_sub == "QUESTION":
                title = "SELECT EXPRESSION TONE"
                t_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
                cv2.putText(frame, title, (cx - t_size[0] // 2, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT_DARK, 2, cv2.LINE_AA)

                draw_pill_button(frame, (cx - 200, btn_y1), (cx - 10, btn_y1 + btn_h), COLOR_TERRACOTTA, "QUESTION", text_color=(255, 255, 255), font_scale=0.55)
                draw_pill_button(frame, (cx + 10, btn_y1), (cx + 200, btn_y1 + btn_h), COLOR_SAND, "SARCASTIC", font_scale=0.55)

        # grammar tense selection modal
        if selecting_tense:
            box_w, box_h = 550, 240
            m_x1, m_y1 = cx - box_w // 2, cy - box_h // 2
            m_x2, m_y2 = cx + box_w // 2, cy + box_h // 2

            draw_rounded_rect(frame, (m_x1, m_y1), (m_x2, m_y2), COLOR_OVERLAY_BG, thickness=-1, radius=16)
            draw_rounded_rect(frame, (m_x1, m_y1), (m_x2, m_y2), COLOR_BORDER, thickness=1, radius=16)

            title = "SELECT TARGET GRAMMAR TENSE"
            t_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
            cv2.putText(frame, title, (cx - t_size[0] // 2, cy - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT_DARK, 2, cv2.LINE_AA)

            sub_txt = f"'{temp_sentence}'"
            s_size = cv2.getTextSize(sub_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            cv2.putText(frame, sub_txt, (cx - s_size[0] // 2, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

            btn_h = 45

            draw_pill_button(frame, (cx - 210, cy - 10), (cx - 10, cy - 10 + btn_h), COLOR_TERRACOTTA, "NOW (-ing)", text_color=(255, 255, 255), font_scale=0.55)
            draw_pill_button(frame, (cx + 10, cy - 10), (cx + 210, cy - 10 + btn_h), COLOR_SAND, "PAST", font_scale=0.55)
            draw_pill_button(frame, (cx - 210, cy + 50), (cx - 10, cy + 50 + btn_h), COLOR_SAND, "FUTURE", font_scale=0.55)
            draw_pill_button(frame, (cx + 10, cy + 50), (cx + 210, cy + 50 + btn_h), COLOR_SAND, "ORIGINAL", font_scale=0.55)

        if is_converting_tense:
            draw_rounded_rect(frame, (cx - 180, cy - 25), (cx + 180, cy + 25), COLOR_BG_CARD, thickness=-1, radius=10)
            draw_rounded_rect(frame, (cx - 180, cy - 25), (cx + 180, cy + 25), COLOR_BORDER, thickness=1, radius=10)
            cv2.putText(frame, "Refining translation with Gemini...", (cx - 150, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_DARK, 1, cv2.LINE_AA)

        # gesture progress bars at frame bottom
        if open_hand_start_time and current_mode == "SPELL":
            hold_elapsed = min(time.time() - open_hand_start_time, TOGGLE_GESTURE_DURATION)
            progress_ratio = hold_elapsed / TOGGLE_GESTURE_DURATION
            action_text = "Stopping..." if is_recording else "Starting..."
            cv2.putText(frame, f"Hold open palm: {action_text}", (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_DARK, 1, cv2.LINE_AA)
            draw_progress_bar(frame, (20, h - 35), (220, h - 23), progress_ratio, color=COLOR_TERRACOTTA)

        if space_start_time:
            space_elapsed = min(time.time() - space_start_time, ACTION_GESTURE_DURATION)
            s_ratio = space_elapsed / ACTION_GESTURE_DURATION
            cv2.putText(frame, "Adding space...", (250, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_DARK, 1, cv2.LINE_AA)
            draw_progress_bar(frame, (250, h - 35), (450, h - 23), s_ratio, color=COLOR_SAGE)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            bg_ai.running = False
            cap.release()
            cv2.destroyAllWindows()
            break
        elif key == ord('m'):
            current_mode = "WORD" if current_mode == "SPELL" else "SPELL"
            selecting_synonym = False
            word_sequence_buffer.clear()
            last_word_pred_time = 0.0
        elif key == ord('s'):
            current_mode = "WORD"
            is_recording = True
            current_word, finished_word = "", ""
            selecting_punctuation, punct_sub, selecting_tense, selecting_synonym = False, None, False, False
            word_sequence_buffer.clear()
            last_word_pred_time = 0.0
        elif key == ord('v'):
            tts.toggle_voice()

if __name__ == "__main__":
    main()