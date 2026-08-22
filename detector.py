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
from collections import deque, Counter
from train_model import extract_hand_features

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
    print("asl_word_model.pkl not found (run train_words.py first if needed):", e)
    asl_word_model = None

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (5, 6), (6, 7), (7, 8),                 # index 
    (9, 10), (10, 11), (11, 12),            # middle 
    (13, 14), (14, 15), (15, 16),           # ring 
    (17, 18), (18, 19), (19, 20),           # pinky
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17) # palm
]

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
        
        self.voices = {
            "Female": "en-US-AriaNeural",
            "Male": "en-US-GuyNeural"
        }
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
                        if tone == "happy":
                            pitch = "+25Hz"
                            rate = "+10%"
                        elif tone == "angry":
                            pitch = "-15Hz"
                            rate = "+15%"
                        elif tone == "surprised":
                            pitch = "+15Hz"
                            rate = "+0%"
                        
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
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                pass
        self.data_queue.put((features, raw_wrist_x)) 

    def run(self):
        prediction_buffer = deque(maxlen=7)
        
        while self.running:
            try:
                item = self.data_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if isinstance(item, tuple):
                features, raw_wrist_x = item
            else:
                features, raw_wrist_x = item, None

            local_letter = "-"
            local_top_candidates = []
                    
            if features is not None and self.asl_model is not None:
                try:
                    if raw_wrist_x is not None:
                        self.wrist_history.append(raw_wrist_x)
                    
                    if hasattr(self.asl_model, "predict_proba"):
                        probs = self.asl_model.predict_proba([features[:123]])[0]
                        classes = self.asl_model.classes_
                        top_idx = np.argsort(probs)[::-1][:2]
                        
                        raw_pred = classes[top_idx[0]]
                        p1, p2 = probs[top_idx[0]], probs[top_idx[1]]

                        #save raw probs for threshold checks
                        local_top_candidates = [
                            (str(classes[top_idx[0]]), float(p1)),
                            (str(classes[top_idx[1]]), float(p2))
                        ]
                    else:
                        raw_pred = self.asl_model.predict([features[:123]])[0]
                        local_top_candidates = [(str(raw_pred), 1.0)]
                    
                    if raw_pred in ['I', 'J'] and len(self.wrist_history) == 15:
                        movement = self.wrist_history[0] - self.wrist_history[-1]
                        if movement > 0.05:
                            raw_pred = 'J'
                        else:
                            raw_pred = 'I'
                            
                    prediction_buffer.append(str(raw_pred))
                    if prediction_buffer:
                        local_letter = Counter(prediction_buffer).most_common(1)[0][0]
                except Exception as e:
                    print("Prediction error:", e)
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
    #check 4 main fingers
    fingers_extended = [
        hand_landmarks[8].y < hand_landmarks[6].y,   # index
        hand_landmarks[12].y < hand_landmarks[10].y, # middle
        hand_landmarks[16].y < hand_landmarks[14].y, # ring
        hand_landmarks[20].y < hand_landmarks[18].y  # pinky
    ]
    
    #check thumb position
    thumb_tip = np.array([hand_landmarks[4].x, hand_landmarks[4].y])
    pinky_mcp = np.array([hand_landmarks[17].x, hand_landmarks[17].y])
    wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y])
    
    thumb_dist = np.linalg.norm(thumb_tip - pinky_mcp)
    hand_scale = np.linalg.norm(wrist - pinky_mcp)
    
    thumb_extended = (thumb_dist > hand_scale * 0.7) or (hand_landmarks[4].y < hand_landmarks[2].y)
    
    return all(fingers_extended) and thumb_extended

def is_two_open_hands(hand1, hand2):
    return is_open_hand(hand1) and is_open_hand(hand2)

def main():
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    
    #lower thresholds slightly so two hands get picked up reliably
    options = vision.HandLandmarkerOptions(
        base_options=base_options, 
        running_mode=vision.RunningMode.VIDEO, 
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55
    )
    detector = vision.HandLandmarker.create_from_options(options)
    
    tts = SpeechEngine()
    bg_ai = BackgroundAI(asl_model)

    cap = WebcamVideoStream(src=0).start()
    time.sleep(1.0)
    
    if not cap.stream.isOpened():
        print("Error: Could not open webcam.")
        return

    window_name = "sign language ai - live classifier"
    
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
    exclamation_sub = False
    selected_tone = "neutral"

    disambiguation_active = False
    disambig_candidates = ("", "")
    disambig_selected = ""
    awaiting_confirmation = False

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
            top_candidates = bg_ai.top_candidates

        #show popup when raw confidence hits 95%+
        if is_recording and not disambiguation_active and len(top_candidates) >= 2:
            (cand1, p1), (cand2, p2) = top_candidates[0], top_candidates[1]
            if (p1 + p2) >= 0.95 and cand1 != cand2:
                disambiguation_active = True
                disambig_candidates = (cand1, cand2)
                awaiting_confirmation = False
                disambig_selected = ""

        if mouse_click_pos is not None:
            mx, my = mouse_click_pos
            mouse_click_pos = None

            if disambiguation_active:
                if not awaiting_confirmation:
                    btn_y1, btn_y2 = cy - 20, cy + 70
                    if (cx - 240) <= mx <= (cx - 20) and btn_y1 <= my <= btn_y2:
                        disambig_selected = disambig_candidates[0]
                        awaiting_confirmation = True
                    elif (cx + 20) <= mx <= (cx + 240) and btn_y1 <= my <= btn_y2:
                        disambig_selected = disambig_candidates[1]
                        awaiting_confirmation = True
                    elif (cx - 100) <= mx <= (cx + 100) and (cy + 85) <= my <= (cy + 120):
                        disambiguation_active = False
                else:
                    btn_y1, btn_y2 = cy + 20, cy + 70
                    if (cx - 210) <= mx <= (cx - 10) and btn_y1 <= my <= btn_y2:
                        current_word += disambig_selected
                        tts.speak(disambig_selected, tone="neutral")
                        disambiguation_active = False
                        awaiting_confirmation = False
                        disambig_selected = ""
                    elif (cx + 10) <= mx <= (cx + 210) and btn_y1 <= my <= btn_y2:
                        disambiguation_active = False
                        awaiting_confirmation = False
                        disambig_selected = ""

            elif (w - 410) <= mx <= (w - 280) and 20 <= my <= 55:
                current_mode = "WORD" if current_mode == "SPELL" else "SPELL"
                word_sequence_buffer.clear()
                last_word_pred_time = 0.0

            elif (w - 140) <= mx <= (w - 20) and 20 <= my <= 55:
                current_word = ""
                finished_word = ""
                selecting_punctuation = False
                exclamation_sub = False

            elif (w - 270) <= mx <= (w - 150) and 20 <= my <= 55:
                current_word = current_word[:-1]

            elif selecting_punctuation:
                btn_y1, btn_y2 = cy - 10, cy + 100

                if not exclamation_sub:
                    btn_w = 200
                    start_x = cx - 320

                    if (start_x) <= mx <= (start_x + btn_w) and btn_y1 <= my <= btn_y2:
                        finished_word = current_word.strip() + "."
                        selected_tone = "neutral"
                        selecting_punctuation = False
                        if finished_word:
                            word_history.append(finished_word)
                            tts.speak(finished_word, selected_tone)

                    elif (start_x + 240) <= mx <= (start_x + 240 + btn_w) and btn_y1 <= my <= btn_y2:
                        exclamation_sub = True

                    elif (start_x + 480) <= mx <= (start_x + 480 + btn_w) and btn_y1 <= my <= btn_y2:
                        finished_word = current_word.strip() + "?"
                        selected_tone = "surprised"
                        selecting_punctuation = False
                        if finished_word:
                            word_history.append(finished_word)
                            tts.speak(finished_word, selected_tone)
                else:
                    btn_w = 220
                    start_x = cx - 240

                    if (start_x) <= mx <= (start_x + btn_w) and btn_y1 <= my <= btn_y2:
                        finished_word = current_word.strip() + "!"
                        selected_tone = "happy"
                        selecting_punctuation = False
                        exclamation_sub = False
                        if finished_word:
                            word_history.append(finished_word)
                            tts.speak(finished_word, selected_tone)

                    elif (start_x + 260) <= mx <= (start_x + 260 + btn_w) and btn_y1 <= my <= btn_y2:
                        finished_word = current_word.strip() + "!"
                        selected_tone = "angry"
                        selecting_punctuation = False
                        exclamation_sub = False
                        if finished_word:
                            word_history.append(finished_word)
                            tts.speak(finished_word, selected_tone)

            elif not is_recording and finished_word:
                btn_y1, btn_y2 = cy + 20, cy + 65

                if (cx - 305) <= mx <= (cx - 115) and btn_y1 <= my <= btn_y2:
                    tts.speak(finished_word, selected_tone)

                elif (cx - 95) <= mx <= (cx + 95) and btn_y1 <= my <= btn_y2:
                    if word_history and word_history[-1] == finished_word:
                        word_history.pop()
                    finished_word = ""
                    selecting_punctuation = True
                    exclamation_sub = False

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

        hand_detected = False
        space_gesture_detected = False
        open_hand = False

        if detection_result.hand_landmarks:
            valid_hands = [h for h in detection_result.hand_landmarks if is_valid_hand_shape(h)]
            
            if valid_hands:
                hand_detected = True
                primary_hand = valid_hands[0]

                curr_wrist = np.array([primary_hand[0].x, primary_hand[0].y])
                if prev_wrist_pos is not None:
                    hand_movement = np.linalg.norm(curr_wrist - prev_wrist_pos)
                else:
                    hand_movement = 0.0
                prev_wrist_pos = curr_wrist

                if len(valid_hands) >= 2:
                    space_gesture_detected = is_two_open_hands(valid_hands[0], valid_hands[1])

                if not space_gesture_detected:
                    open_hand = is_open_hand(primary_hand)

                if is_recording and space_gesture_detected and not disambiguation_active:
                    if space_start_time is None:
                        space_start_time = time.time()
                    elif (time.time() - space_start_time >= ACTION_GESTURE_DURATION) and not space_triggered:
                        current_word += " "
                        space_triggered = True
                else:
                    space_start_time = None
                    space_triggered = False

                if open_hand and not space_gesture_detected and hand_movement < 0.035 and not disambiguation_active:
                    if open_hand_start_time is None:
                        open_hand_start_time = time.time()
                    elif (time.time() - open_hand_start_time >= TOGGLE_GESTURE_DURATION) and not open_hand_triggered:
                        open_hand_triggered = True
                        is_recording = not is_recording
                        
                        if is_recording:
                            current_word = ""
                            finished_word = ""
                            selecting_punctuation = False
                            exclamation_sub = False
                            word_sequence_buffer.clear()
                            last_word_pred_time = 0.0
                        else:
                            if current_word.strip():
                                selecting_punctuation = True
                                exclamation_sub = False
                            else:
                                finished_word = ""
                        open_hand_start_time = None
                else:
                    open_hand_start_time = None
                    open_hand_triggered = False

                pts = np.array([[lm.x, lm.y, lm.z] for lm in primary_hand])
                features = extract_hand_features(pts)
                
                bg_ai.update_data(features, primary_hand[0].x)

                for hand_landmarks in valid_hands:
                    for connection in HAND_CONNECTIONS:
                        start_p = (int(hand_landmarks[connection[0]].x * w), int(hand_landmarks[connection[0]].y * h))
                        end_p = (int(hand_landmarks[connection[1]].x * w), int(hand_landmarks[connection[1]].y * h))
                        cv2.line(frame, start_p, end_p, (255, 255, 255), 2)

                    for landmark in hand_landmarks:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            else:
                bg_ai.update_data(None, None)
                open_hand_start_time = None
                space_start_time = None
                letter_hold_start_time = None
                current_holding_letter = None
                prev_wrist_pos = None
        else:
            bg_ai.update_data(None, None)
            open_hand_start_time = None
            space_start_time = None
            letter_hold_start_time = None
            current_holding_letter = None
            prev_wrist_pos = None

        if is_recording and not open_hand and not space_gesture_detected and hand_detected and not disambiguation_active:
            if current_mode == "SPELL":
                if predicted_letter == current_holding_letter:
                    elapsed = time.time() - letter_hold_start_time
                    if elapsed >= HOLD_LETTER_DURATION:
                        current_word += predicted_letter
                        tts.speak(predicted_letter, tone="neutral")
                        letter_hold_start_time = time.time()  
                else:
                    current_holding_letter = predicted_letter
                    letter_hold_start_time = time.time()

            elif current_mode == "WORD" and asl_word_model is not None:
                if time.time() - last_word_pred_time >= WORD_COOLDOWN_DURATION:
                    word_sequence_buffer.append(features)

                    if len(word_sequence_buffer) == 30:
                        flat_sequence = np.array(word_sequence_buffer).flatten()
                        try:
                            predicted_word = asl_word_model.predict([flat_sequence])[0]
                            if predicted_word:
                                current_word += f"{predicted_word} "
                                tts.speak(predicted_word, tone=selected_tone)
                        except Exception as e:
                            print("Word prediction error:", e)

                        word_sequence_buffer.clear()
                        last_word_pred_time = time.time()
                else:
                    word_sequence_buffer.clear()
        else:
            letter_hold_start_time = None
            current_holding_letter = None

        #draw hud
        mode_color = (0, 165, 255) if current_mode == "SPELL" else (255, 0, 150)
        cv2.rectangle(frame, (w - 410, 20), (w - 280, 55), mode_color, -1)
        cv2.rectangle(frame, (w - 410, 20), (w - 280, 55), (255, 255, 255), 2)
        cv2.putText(frame, f"MODE: {current_mode}", (w - 400, 43), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.rectangle(frame, (w - 140, 20), (w - 20, 55), (0, 0, 200), -1)
        cv2.rectangle(frame, (w - 140, 20), (w - 20, 55), (255, 255, 255), 2)
        cv2.putText(frame, "CLEAR", (w - 110, 43), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.rectangle(frame, (w - 270, 20), (w - 150, 55), (0, 100, 200), -1)
        cv2.rectangle(frame, (w - 270, 20), (w - 150, 55), (255, 255, 255), 2)
        cv2.putText(frame, "DELETE", (w - 245, 43), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        box_x1, box_y1 = w - 300, 75
        box_x2, box_y2 = w - 20, 260 
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (30, 30, 30), -1)
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), 1)

        instructions = [
            "CONTROLS:",
            "Toggle Mode: Click MODE or press 'm'",
            "Start/Finish: Hold STILL open palm up",
            "Space: Hold 2 open palms facing screen",
            "Delete: Click 'DELETE' button",
            "Clear All: Click 'CLEAR' button",
            "Type Letter: Hold sign for 2s",
            "Press 'v' to toggle Voice",
            "Press 'q' to quit."
        ]

        for idx, line_text in enumerate(instructions):
            y_pos = box_y1 + 18 + (idx * 20)
            text_color = (0, 255, 255) if idx == 0 else (255, 255, 255)
            cv2.putText(frame, line_text, (box_x1 + 10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1, cv2.LINE_AA)

        cv2.putText(frame, f"sign: {predicted_letter}", (40, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"voice: {tts.current_voice_label}", (40, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 150, 0), 2, cv2.LINE_AA)

        if is_recording:
            cv2.putText(frame, f"[recording mode - {current_mode}]", (40, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"word: {current_word}_", (40, 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 255), 3, cv2.LINE_AA)

        if disambiguation_active:
            box_w, box_h = 600, 260
            m_x1, m_y1 = cx - box_w // 2, cy - box_h // 2
            m_x2, m_y2 = cx + box_w // 2, cy + box_h // 2

            cv2.rectangle(frame, (m_x1, m_y1), (m_x2, m_y2), (20, 20, 20), -1)
            cv2.rectangle(frame, (m_x1, m_y1), (m_x2, m_y2), (0, 215, 255), 2)

            if not awaiting_confirmation:
                title = "Algorithm confused! Did you mean:"
                t_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.putText(frame, title, (cx - t_size[0] // 2, cy - 65), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2, cv2.LINE_AA)

                btn_w, btn_h = 220, 90
                btn_y1 = cy - 20

                c1_text = disambig_candidates[0]
                cv2.rectangle(frame, (cx - 240, btn_y1), (cx - 20, btn_y1 + btn_h), (180, 100, 0), -1)
                cv2.rectangle(frame, (cx - 240, btn_y1), (cx - 20, btn_y1 + btn_h), (255, 255, 255), 2)
                cv2.putText(frame, c1_text, (cx - 140, btn_y1 + 55), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

                c2_text = disambig_candidates[1]
                cv2.rectangle(frame, (cx + 20, btn_y1), (cx + 240, btn_y1 + btn_h), (0, 140, 200), -1)
                cv2.rectangle(frame, (cx + 20, btn_y1), (cx + 240, btn_y1 + btn_h), (255, 255, 255), 2)
                cv2.putText(frame, c2_text, (cx + 120, btn_y1 + 55), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

                cv2.rectangle(frame, (cx - 100, cy + 85), (cx + 100, cy + 120), (80, 80, 80), -1)
                cv2.putText(frame, "DISMISS", (cx - 45, cy + 108), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                title = f"Are you sure you want to add '{disambig_selected}'?"
                t_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0]
                cv2.putText(frame, title, (cx - t_size[0] // 2, cy - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

                btn_y1, btn_y2 = cy + 20, cy + 70

                cv2.rectangle(frame, (cx - 210, btn_y1), (cx - 10, btn_y2), (0, 160, 0), -1)
                cv2.rectangle(frame, (cx - 210, btn_y1), (cx - 10, btn_y2), (255, 255, 255), 2)
                cv2.putText(frame, "YES, ADD IT", (cx - 170, btn_y1 + 32), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.rectangle(frame, (cx + 10, btn_y1), (cx + 210, btn_y2), (0, 0, 180), -1)
                cv2.rectangle(frame, (cx + 10, btn_y1), (cx + 210, btn_y2), (255, 255, 255), 2)
                cv2.putText(frame, "CANCEL", (cx + 70, btn_y1 + 32), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        #progress bar render logic
        if open_hand_start_time:
            hold_elapsed = min(time.time() - open_hand_start_time, TOGGLE_GESTURE_DURATION)
            progress_ratio = hold_elapsed / TOGGLE_GESTURE_DURATION
            action_text = "finishing..." if is_recording else "starting..."
            cv2.putText(frame, f"hold open hand: {action_text}", (40, h - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (40, h - 35), (240, h - 20), (100, 100, 100), 2)
            cv2.rectangle(frame, (40, h - 35), (40 + int(200 * progress_ratio), h - 20), (0, 255, 255), -1)

        if space_start_time:
            space_elapsed = min(time.time() - space_start_time, ACTION_GESTURE_DURATION)
            s_ratio = space_elapsed / ACTION_GESTURE_DURATION
            cv2.putText(frame, "adding space...", (280, h - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (280, h - 35), (480, h - 20), (100, 100, 100), 2)
            cv2.rectangle(frame, (280, h - 35), (280 + int(200 * s_ratio), h - 20), (255, 255, 0), -1)

        if is_recording and current_mode == "SPELL" and current_holding_letter and current_holding_letter != "-" and letter_hold_start_time:
            letter_elapsed = min(time.time() - letter_hold_start_time, HOLD_LETTER_DURATION)
            l_ratio = letter_elapsed / HOLD_LETTER_DURATION
            cv2.putText(frame, f"typing '{current_holding_letter}'...", (w - 280, h - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (w - 280, h - 35), (w - 40, h - 20), (100, 100, 100), 2)
            cv2.rectangle(frame, (w - 280, h - 35), (w - 280 + int(240 * l_ratio), h - 20), (0, 255, 0), -1)

        if is_recording and current_mode == "WORD":
            buf_len = len(word_sequence_buffer)
            w_ratio = buf_len / 30.0
            cv2.putText(frame, f"word buffer: {buf_len}/30 frames", (w - 320, h - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 150), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (w - 320, h - 35), (w - 40, h - 20), (100, 100, 100), 2)
            cv2.rectangle(frame, (w - 320, h - 35), (w - 320 + int(280 * w_ratio), h - 20), (255, 0, 150), -1)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            bg_ai.running = False
            cap.stop()
            cv2.destroyAllWindows()
            break
        elif key == ord('m'):
            current_mode = "WORD" if current_mode == "SPELL" else "SPELL"
            word_sequence_buffer.clear()
            last_word_pred_time = 0.0
        elif key == ord('v'):
            tts.toggle_voice()

if __name__ == "__main__":
    main()