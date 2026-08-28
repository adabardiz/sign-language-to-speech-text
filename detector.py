import cv2
import numpy as np
import mediapipe as mp
import joblib
import time
import threading
import queue
import sys
import os
import asyncio
import edge_tts
import pygame
import urllib.request
import json
from collections import deque, Counter
from train_model import extract_hand_features

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# warm minimalist color palette (bgr format)
COLOR_BG_CARD    = (235, 238, 240)  
COLOR_TEXT_DARK  = (42, 42, 42)     
COLOR_TEXT_MUTED = (120, 120, 120)  
COLOR_TERRACOTTA = (60, 110, 195)   
COLOR_SAGE       = (120, 160, 100)  
COLOR_ROSE       = (90, 90, 200)    
COLOR_SAND       = (210, 215, 220)  
COLOR_BORDER     = (180, 185, 190)  
COLOR_OVERLAY_BG = (245, 247, 248)  

# ui drawing helper functions
def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=10):
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1
    r = min(radius, w // 2, h // 2)

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

def draw_pill_button(img, pt1, pt2, bg_color, text, text_color=COLOR_TEXT_DARK, font_scale=0.5, radius=8):
    draw_rounded_rect(img, pt1, pt2, bg_color, thickness=-1, radius=radius)
    draw_rounded_rect(img, pt1, pt2, COLOR_BORDER, thickness=1, radius=radius)
    
    x1, y1 = pt1
    x2, y2 = pt2
    w, h = x2 - x1, y2 - y1
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
    tx = x1 + (w - tw) // 2
    ty = y1 + (h + th) // 2 - 1
    cv2.putText(img, text, (tx, ty), font, font_scale, text_color, 1, cv2.LINE_AA)

def draw_progress_bar(img, pt1, pt2, progress_ratio, color=COLOR_TERRACOTTA):
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1
    
    # track background
    draw_rounded_rect(img, (x1, y1), (x2, y2), COLOR_SAND, thickness=-1, radius=h // 2)
    
    # filled progress
    if progress_ratio > 0.01:
        fill_w = int(w * min(max(progress_ratio, 0.0), 1.0))
        if fill_w > h:
            draw_rounded_rect(img, (x1, y1), (x1 + fill_w, y2), color, thickness=-1, radius=h // 2)

# model & feature extraction setup
KEY_FACE_INDICES = [
    1,                  # nose tip anchor
    33, 133, 159, 145,  # left eye
    362, 263, 386, 374, # right eye
    70, 63, 105, 66,    # left eyebrow
    300, 293, 334, 296, # right eyebrow
    61, 291, 0, 17, 13, 14, # outer & inner mouth
    78, 308, 82, 312    # lip curves
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

def aggregate_sequence(sequence_matrix):
    seq = np.array(sequence_matrix, dtype=np.float32)
    mean_f = np.mean(seq, axis=0)
    std_f = np.std(seq, axis=0)
    delta_f = seq[-1] - seq[0]
    max_f = np.max(seq, axis=0)
    min_f = np.min(seq, axis=0)
    return np.hstack([mean_f, std_f, delta_f, max_f, min_f])

try:
    asl_model = joblib.load('asl_model.pkl')
    print("Loaded ASL alphabet model successfully.")
except Exception as e:
    print("Error loading alphabet model:", e)
    asl_model = None

try:
    asl_word_model = joblib.load('asl_word_model.pkl', mmap_mode=None)
    print("Loaded ASL word model successfully.")
except Exception as e:
    print("asl_word_model.pkl not found:", e)
    asl_word_model = None

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (5, 6), (6, 7), (7, 8),                 # index
    (9, 10), (10, 11), (11, 12),            # middle
    (13, 14), (14, 15), (15, 16),           # ring
    (17, 18), (18, 19), (19, 20),           # pinky
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17) # palm
]

def transform_tense(text, target_tense):
    if not text or target_tense == "ORIGINAL":
        return text

    punct = ""
    if text[-1] in ".!?":
        punct = text[-1]
        clean = text[:-1].strip()
    else:
        clean = text.strip()

    gemini_key = os.environ.get("GEMINI_API_KEY")

    if gemini_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
            
            tense_instructions = {
                "NOW": "present continuous tense (e.g., 'I am going', 'She is eating')",
                "PAST": "simple past tense (e.g., 'I went', 'She ate')",
                "FUTURE": "future tense using 'will' or 'going to' (e.g., 'I will go')",
                "PRESENT": "simple present tense (e.g., 'I go', 'She eats')"
            }
            target_desc = tense_instructions.get(target_tense, target_tense.lower())

            prompt = (
                f"You are a sign language translator. Convert this raw ASL gloss/phrase into fluent, "
                f"grammatically correct English in the {target_desc}.\n"
                f"Guidelines:\n"
                f"- Fix missing prepositions, articles (a, an, the), and ASL word order.\n"
                f"- Do not add extra commentary or explanation.\n"
                f"- Output ONLY the converted sentence.\n\n"
                f"ASL Input: '{clean}'"
            )

            payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode('utf-8')
            req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=3) as resp:
                res = json.loads(resp.read().decode('utf-8'))
                out = res["candidates"][0]["content"]["parts"][0]["text"].strip()
                if punct and not out[-1] in ".!?":
                    out += punct
                return out
        except Exception as e:
            print("LLM Tense conversion api error:", e)

    words = clean.lower().split()
    if not words:
        return text

    pronouns = {"i", "you", "he", "she", "we", "they", "it"}
    first = words[0]
    rest = " ".join(words[1:]) if len(words) > 1 else ""

    if target_tense == "NOW":
        aux = "am" if first == "i" else ("is" if first in ["he", "she", "it"] else "are")
        res = f"{words[0]} {aux} {rest}".strip() if first in pronouns else f"is {clean}".strip()
    elif target_tense == "FUTURE":
        res = f"{words[0]} will {rest}".strip() if first in pronouns else f"will {clean}".strip()
    elif target_tense == "PAST":
        res = f"{words[0]} went/did {rest}".strip() if first in pronouns else f"did {clean}".strip()
    else:
        res = clean

    if punct and not res.endswith(punct):
        res += punct
    return res.capitalize()

class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.grabbed, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

def is_valid_hand_shape(landmarks):
    wrist = np.array([landmarks[0].x, landmarks[0].y])
    middle_mcp = np.array([landmarks[9].x, landmarks[9].y])
    palm_size = np.linalg.norm(wrist - middle_mcp)
    return 0.03 < palm_size < 0.40

class SpeechEngine:
    def __init__(self):
        self.speech_queue = queue.Queue()
        pygame.mixer.init()
        self.cache_dir = "tts_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.voices = {"Female": "en-US-AriaNeural", "Male": "en-US-GuyNeural"}
        self.current_voice_label = "Female"
        
        self.thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.thread.start()

    def toggle_voice(self):
        self.current_voice_label = "Male" if self.current_voice_label == "Female" else "Female"

    def _speech_worker(self):
        while True:
            task = self.speech_queue.get()
            if task:
                text, voice_label, tone = task
                clean_text = str(text).lower().strip()
                if clean_text:
                    try:
                        safe_filename = "".join([c for c in clean_text if c.isalnum()]) or "speech"
                        pitch = "+0Hz"
                        rate = "+0%"
                        if tone == "happy": pitch, rate = "+25Hz", "+10%"
                        elif tone == "angry": pitch, rate = "-15Hz", "+15%"
                        elif tone == "surprised": pitch, rate = "+15Hz", "+0%"
                        elif tone == "sarcastic": pitch, rate = "-20Hz", "-25%"
                        elif tone == "sad": pitch, rate = "-15Hz", "-30%"
                        
                        cache_file = os.path.join(self.cache_dir, f"{safe_filename}_{voice_label}_{tone}.mp3")

                        if not os.path.exists(cache_file):
                            voice_id = self.voices[voice_label]
                            communicate = edge_tts.Communicate(clean_text, voice_id, pitch=pitch, rate=rate)
                            asyncio.run(communicate.save(cache_file))

                        pygame.mixer.music.load(cache_file)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.01)
                        pygame.mixer.music.unload()
                    except Exception as e:
                        print("Speech engine error:", e)
            self.speech_queue.task_done()

    def speak(self, text, tone="neutral"):
        if text and str(text).strip():
            self.speech_queue.put((str(text).strip(), self.current_voice_label, tone))

class BackgroundAI:
    def __init__(self, model):
        self.asl_model = model
        self.data_queue = queue.Queue(maxsize=1)
        self.result_lock = threading.Lock()
        
        self.predicted_letter = "-"
        self.top_candidates = []
        self.wrist_history = deque(maxlen=15)
        
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def update_data(self, features, raw_wrist_x=None):
        if not self.data_queue.empty():
            try: self.data_queue.get_nowait()
            except queue.Empty: pass
        self.data_queue.put((features, raw_wrist_x)) 

    def run(self):
        prediction_buffer = deque(maxlen=7)
        while self.running:
            try:
                item = self.data_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            features, raw_wrist_x = item if isinstance(item, tuple) else (item, None)
            local_letter = "-"
            local_top_candidates = []
                    
            if features is not None and self.asl_model is not None:
                try:
                    if raw_wrist_x is not None:
                        self.wrist_history.append(raw_wrist_x)
                    
                    if hasattr(self.asl_model, "predict_proba"):
                        probs = self.asl_model.predict_proba([features])[0]
                        classes = self.asl_model.classes_
                        top_idx = np.argsort(probs)[::-1][:2]
                        raw_pred = classes[top_idx[0]]
                        local_top_candidates = [(str(classes[top_idx[0]]), float(probs[top_idx[0]])),
                                                (str(classes[top_idx[1]]), float(probs[top_idx[1]]))]
                    else:
                        raw_pred = self.asl_model.predict([features])[0]
                        local_top_candidates = [(str(raw_pred), 1.0)]
                    
                    if raw_pred in ['I', 'J'] and len(self.wrist_history) == 15:
                        raw_pred = 'J' if (self.wrist_history[0] - self.wrist_history[-1]) > 0.05 else 'I'
                            
                    prediction_buffer.append(str(raw_pred))
                    if prediction_buffer:
                        local_letter = Counter(prediction_buffer).most_common(1)[0][0]
                except Exception as e:
                    local_letter = "-"
                    local_top_candidates = []
            else:
                prediction_buffer.clear()
                self.wrist_history.clear()
                local_letter = "-"
                local_top_candidates = []

            with self.result_lock:
                self.predicted_letter = local_letter
                self.top_candidates = local_top_candidates

def is_open_hand(hand_landmarks):
    fingers_extended = [
        hand_landmarks[8].y < hand_landmarks[6].y,
        hand_landmarks[12].y < hand_landmarks[10].y,
        hand_landmarks[16].y < hand_landmarks[14].y,
        hand_landmarks[20].y < hand_landmarks[18].y
    ]
    thumb_tip = np.array([hand_landmarks[4].x, hand_landmarks[4].y])
    pinky_mcp = np.array([hand_landmarks[17].x, hand_landmarks[17].y])
    wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y])
    
    thumb_dist = np.linalg.norm(thumb_tip - pinky_mcp)
    hand_scale = np.linalg.norm(wrist - pinky_mcp)
    thumb_extended = (thumb_dist > hand_scale * 0.7) or (hand_landmarks[4].y < hand_landmarks[2].y)
    
    return all(fingers_extended) and thumb_extended

def is_two_open_hands(hand1, hand2):
    return is_open_hand(hand1) and is_open_hand(hand2)

# main execution loop
def main():
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
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55
    )

    tts = SpeechEngine()
    bg_ai = BackgroundAI(asl_model)

    cap = WebcamVideoStream(src=0).start()
    time.sleep(1.0)
    
    if not cap.stream.isOpened():
        print("Error: Could not open webcam.")
        return

    window_name = "Sign Language Translator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    mouse_click_pos = None
    def on_mouse(event, x, y, flags, param):
        nonlocal mouse_click_pos
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_click_pos = (x, y)

    cv2.setMouseCallback(window_name, on_mouse)

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

        # vision pipeline & gesture recognition
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
                        tts.speak(predicted_letter, tone="neutral")
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
                                if "/" in predicted_word:
                                    synonym_options = [w.strip() for w in predicted_word.split("/") if w.strip()]
                                    selecting_synonym = True
                                else:
                                    current_word += f"{predicted_word} "
                                    tts.speak(predicted_word, tone=selected_tone)
                        except Exception as e:
                            print("Word prediction error:", e)

                        word_sequence_buffer.clear()
                        last_word_pred_time = time.time()
                else:
                    word_sequence_buffer.clear()
        else:
            letter_hold_start_time, current_holding_letter = None, None


        # top control buttons
        rec_bg = COLOR_SAGE if not is_recording else COLOR_ROSE
        rec_txt_color = (255, 255, 255)
        rec_btn_text = "STOP" if is_recording else "START"
        draw_pill_button(frame, (w - 550, 20), (w - 420, 55), rec_bg, rec_btn_text, text_color=rec_txt_color, font_scale=0.55)

        mode_bg = COLOR_TERRACOTTA if current_mode == "SPELL" else COLOR_SAND
        mode_txt_color = (255, 255, 255) if current_mode == "SPELL" else COLOR_TEXT_DARK
        draw_pill_button(frame, (w - 410, 20), (w - 280, 55), mode_bg, f"MODE: {current_mode}", text_color=mode_txt_color, font_scale=0.45)

        draw_pill_button(frame, (w - 270, 20), (w - 150, 55), COLOR_SAND, "DELETE", text_color=COLOR_TEXT_DARK, font_scale=0.55)
        draw_pill_button(frame, (w - 140, 20), (w - 20, 55), COLOR_ROSE, "CLEAR", text_color=(255, 255, 255), font_scale=0.55)

        # controls panel
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

        draw_rounded_rect(frame, (20, 20), (220, 65), COLOR_BG_CARD, thickness=-1, radius=10)
        draw_rounded_rect(frame, (20, 20), (220, 65), COLOR_BORDER, thickness=1, radius=10)
        cv2.putText(frame, "CURRENT SIGN", (32, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{predicted_letter}", (32, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT_DARK, 2, cv2.LINE_AA)

        # floating voice indicator card
        draw_rounded_rect(frame, (230, 20), (430, 65), COLOR_BG_CARD, thickness=-1, radius=10)
        draw_rounded_rect(frame, (230, 20), (430, 65), COLOR_BORDER, thickness=1, radius=10)
        cv2.putText(frame, "VOICE ENGINE", (242, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{tts.current_voice_label}", (242, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TERRACOTTA, 2, cv2.LINE_AA)

        if is_recording:
            draw_rounded_rect(frame, (20, 80), (430, 160), COLOR_BG_CARD, thickness=-1, radius=12)
            draw_rounded_rect(frame, (20, 80), (430, 160), COLOR_BORDER, thickness=1, radius=12)
            
            cv2.putText(frame, f"RECORDING ({current_mode})", (32, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_SAGE, 2, cv2.LINE_AA)
            
            disp_word = f"{current_word}_" if current_word else "..."
            cv2.putText(frame, disp_word, (32, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT_DARK, 2, cv2.LINE_AA)

        # holds / progress indicator bars
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


        # synonym overlay modal
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

        # punctuation & tone selection modal
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

        # tense selection modal
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

        # gestural hold indicators (palm toggle / space)
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

        # final render display
        cv2.imshow(window_name, frame)

        # keyboard shortcuts
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            bg_ai.running = False
            cap.stop()
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