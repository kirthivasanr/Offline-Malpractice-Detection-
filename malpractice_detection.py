import sys
print(f"!!! Running Detection Script with: {sys.executable}")
import cv2
import numpy as np
import mediapipe as mp
import os
import time
from datetime import datetime
from collections import deque
import math

class MalpracticeDetector:
    def __init__(self, camera_id=0, output_folder="violations"):
        self.camera_id = camera_id
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        self.violation_types = ["peeking", "hand_signs", "passing_object"]
        for vt in self.violation_types:
            os.makedirs(os.path.join(self.output_folder, vt), exist_ok=True)

        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec_mesh = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 200, 0))
        self.drawing_spec_hands = self.mp_drawing.DrawingSpec(thickness=1, color=(0, 0, 255))

        # Detection parameters
        self.violation_cooldown = 5.0
        self.last_violation_time = {vt: 0.0 for vt in self.violation_types}

        # --- Peeking Detection --- (Keep as is from last version)
        self.peeking_buffer_deviation = deque(maxlen=10)
        self.peeking_buffer_translation = deque(maxlen=10)
        self.nose_pos_buffer = deque(maxlen=5)
        self.peeking_deviation_threshold_strict = 0.65
        self.peeking_translation_threshold_px = 25
        self.peeking_sustained_frames = 6
        self.LM_LEFT_FACE_EDGE = 234; self.LM_RIGHT_FACE_EDGE = 454; self.LM_NOSE_TIP = 1

        # --- Hand Sign Detection --- (Keep as is)
        self.hand_gesture_buffer = {'left': deque(maxlen=15), 'right': deque(maxlen=15)}
        self.hand_gesture_threshold = 10
        self.suspicious_gestures = {'ONE_FINGER', 'TWO_FINGERS', 'THREE_FINGERS', 'FIST'}

        # --- Passing Object Detection (Refined Motion Check) ---
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)
        self.object_contour_buffer = deque(maxlen=8) # Buffer to check stability

        # *** Object Tuning Parameters ***
        # Size limits for foreground blobs to be considered objects
        self.object_area_min = 400       # TUNE: Increase if missing small objects, decrease if catching noise (300-800)
        self.object_area_max = 7000      # TUNE: Decrease to exclude body parts, increase *cautiously* if missing larger objects (4000-10000)
        # Shape limits
        self.object_aspect_ratio_min = 0.2 # Allows thin objects (phones on side)
        self.object_aspect_ratio_max = 5.0 # Allows wide objects (notes) - Adjust if needed
        # Overlap limits (how much % of object area can overlap before being ignored)
        self.face_overlap_threshold = 0.25  # TUNE: Increase if face blocks object, decrease if detecting face parts (0.1-0.4)
        self.hand_overlap_threshold = 0.45  # TUNE: Increase if hands block object, decrease if detecting hands (0.3-0.6)
        # Stability and Motion
        self.object_stability_threshold = 4  # TUNE: How many frames (out of buffer len) object needs to be stable (3-6)
        self.object_min_stable_motion_px = 8 # TUNE: Min distance centroid must move during stable window to be non-static (5-15 pixels)


        # System state
        self.cap = None; self.running = False; self.frame_width = 640; self.frame_height = 480
        self.current_hand_bboxes = []; self.current_face_bbox = None


    def start(self): # Same
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened(): print(f"ERROR: Cannot open camera ID {self.camera_id}"); return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)); self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened ({self.frame_width}x{self.frame_height})."); self.running = True
        self.run_detection_loop(); return True

    def stop(self): # Same
        self.running = False
        if self.cap: self.cap.release()
        cv2.destroyAllWindows(); print("Detection stopped.")
        if hasattr(self, 'face_mesh') and self.face_mesh: self.face_mesh.close()
        if hasattr(self, 'hands') and self.hands: self.hands.close()


    def estimate_head_yaw_simple(self, face_landmarks): # Same
        deviation_score = 0.0; nose_pos = None
        if not face_landmarks: return deviation_score, nose_pos
        lm = face_landmarks.landmark; img_w = self.frame_width
        try:
            left_edge_x = lm[self.LM_LEFT_FACE_EDGE].x * img_w; right_edge_x = lm[self.LM_RIGHT_FACE_EDGE].x * img_w
            nose_x = lm[self.LM_NOSE_TIP].x * img_w; nose_y = lm[self.LM_NOSE_TIP].y * self.frame_height
            nose_pos = (int(nose_x), int(nose_y))
        except IndexError: print("WARN: Landmark index out of bounds."); return deviation_score, nose_pos
        dist_left = nose_x - left_edge_x; dist_right = right_edge_x - nose_x
        if dist_left <= 1 or dist_right <= 1: return deviation_score, nose_pos
        deviation_score = abs(dist_left - dist_right) / (dist_left + dist_right)
        return deviation_score, nose_pos

    def calculate_translation(self): # Same
        if len(self.nose_pos_buffer) < 2: return 0.0
        start_pos = self.nose_pos_buffer[0]; end_pos = self.nose_pos_buffer[-1]
        delta_x = end_pos[0] - start_pos[0]; delta_y = end_pos[1] - start_pos[1]
        return math.sqrt(delta_x**2 + delta_y**2)

    def detect_peeking(self, frame, face_results): # Same logic
        deviation_score = 0.0; nose_pos = None; translation_magnitude = 0.0
        self.current_face_bbox = None
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                deviation_score, nose_pos = self.estimate_head_yaw_simple(face_landmarks)
                h, w, _ = frame.shape; x_coords = [lm.x * w for lm in face_landmarks.landmark]; y_coords = [lm.y * h for lm in face_landmarks.landmark]
                x_min = int(min(x_coords)); y_min = int(min(y_coords)); x_max = int(max(x_coords)); y_max = int(max(y_coords))
                # Reduced padding slightly for face bbox used in object filtering
                padding = 3
                self.current_face_bbox = (max(0, x_min - padding), max(0, y_min - padding), min(w, x_max + padding), min(h, y_max + padding))
                # Optional: Draw face bbox
                # cv2.rectangle(frame, (self.current_face_bbox[0], self.current_face_bbox[1]), (self.current_face_bbox[2], self.current_face_bbox[3]), (0, 255, 255), 1)
                break
        if nose_pos: self.nose_pos_buffer.append(nose_pos)
        translation_magnitude = self.calculate_translation()
        self.peeking_buffer_deviation.append(deviation_score); self.peeking_buffer_translation.append(translation_magnitude)
        sustained_deviation_count = sum(1 for score in self.peeking_buffer_deviation if score > self.peeking_deviation_threshold_strict)
        sustained_translation_count = sum(1 for trans in self.peeking_buffer_translation if trans > self.peeking_translation_threshold_px)
        is_peeking = False; violation_reason = ""
        if sustained_deviation_count >= self.peeking_sustained_frames: is_peeking = True; violation_reason = f"Rotation (Dev:{deviation_score:.2f})"
        elif sustained_translation_count >= self.peeking_sustained_frames: is_peeking = True; violation_reason = f"Translation (Dist:{translation_magnitude:.1f}px)"
        status_text = f"Head Dev: {deviation_score:.2f} | Trans: {translation_magnitude:.1f}px"; color = (0, 255, 0)
        if is_peeking:
            status_text += " - PEEKING?"; color = (0, 0, 255)
            current_time = time.time()
            if current_time - self.last_violation_time["peeking"] > self.violation_cooldown:
                self.last_violation_time["peeking"] = current_time; details = {"reason": violation_reason, "deviation": deviation_score, "translation": translation_magnitude}
                self.save_violation(frame, "peeking", details); print(f"VIOLATION: Peeking detected ({violation_reason})")
        cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    def get_hand_gesture(self, hand_landmarks, handedness): # No change
        if not hand_landmarks: return "NO_HAND"
        lm = hand_landmarks.landmark; label = handedness.classification[0].label; fingers_extended = []
        thumb_tip = lm[self.mp_hands.HandLandmark.THUMB_TIP]; thumb_ip = lm[self.mp_hands.HandLandmark.THUMB_IP]; wrist = lm[self.mp_hands.HandLandmark.WRIST]
        if math.dist((thumb_tip.x, thumb_tip.y), (wrist.x, wrist.y)) > math.dist((thumb_ip.x, thumb_ip.y), (wrist.x, wrist.y)): fingers_extended.append("THUMB")
        if lm[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y: fingers_extended.append("INDEX")
        if lm[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y: fingers_extended.append("MIDDLE")
        if lm[self.mp_hands.HandLandmark.RING_FINGER_TIP].y < lm[self.mp_hands.HandLandmark.RING_FINGER_PIP].y: fingers_extended.append("RING")
        if lm[self.mp_hands.HandLandmark.PINKY_TIP].y < lm[self.mp_hands.HandLandmark.PINKY_PIP].y: fingers_extended.append("PINKY")
        num_fingers = len(fingers_extended)
        if num_fingers == 0:
             palm_center_x=lm[self.mp_hands.HandLandmark.WRIST].x; palm_center_y=(lm[self.mp_hands.HandLandmark.WRIST].y+lm[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y)/2; is_fist=True
             for tip_lm in [self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.PINKY_TIP]:
                 pip_lm=tip_lm-1; tip_dist=math.dist((lm[tip_lm].x,lm[tip_lm].y),(palm_center_x,palm_center_y)); pip_dist=math.dist((lm[pip_lm].x,lm[pip_lm].y),(palm_center_x,palm_center_y))
                 if tip_dist > pip_dist: is_fist=False; break
             if is_fist: return "FIST"; 
             else: return "ZERO_FINGERS"
        elif num_fingers == 1: return "ONE_FINGER"; 
        elif num_fingers == 2: return "TWO_FINGERS"; 
        elif num_fingers == 3: return "THREE_FINGERS"; 
        elif num_fingers == 4: return "FOUR_FINGERS"; 
        elif num_fingers == 5: return "FIVE_FINGERS"; 
        else: return "UNKNOWN"

    def detect_hand_signs(self, frame, hand_results): # Same logic
        detected_gestures = {'left': "NO_HAND", 'right': "NO_HAND"}; self.current_hand_bboxes = []
        if hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                handedness = hand_results.multi_handedness[idx]; label = handedness.classification[0].label.lower()
                gesture = self.get_hand_gesture(hand_landmarks, handedness); detected_gestures[label] = gesture
                h, w, _ = frame.shape; x_coords = [lm.x * w for lm in hand_landmarks.landmark]; y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(x_coords)), int(max(x_coords)); y_min, y_max = int(min(y_coords)), int(max(y_coords))
                padding = 10; x_min=max(0,x_min-padding); y_min=max(0,y_min-padding); x_max=min(w,x_max+padding); y_max=min(h,y_max+padding)
                self.current_hand_bboxes.append((x_min, y_min, x_max, y_max))
        self.hand_gesture_buffer['left'].append(detected_gestures['left']); self.hand_gesture_buffer['right'].append(detected_gestures['right'])
        violation_found = False; violation_details = {}
        for hand_label in ['left', 'right']:
            buffer = self.hand_gesture_buffer[hand_label];
            if not buffer: continue
            counts = {gest: buffer.count(gest) for gest in self.suspicious_gestures}
            for gesture, count in counts.items():
                if count >= self.hand_gesture_threshold: violation_found=True; violation_details={"hand":hand_label, "gesture":gesture, "count":count}; break
            if violation_found: break
        status_l = f"L: {detected_gestures['left']}"; status_r = f"R: {detected_gestures['right']}"; color = (0, 255, 0)
        if violation_found:
            color = (0, 0, 255); status_l += f" ({violation_details.get('hand', '')} V)"; status_r += f" ({violation_details.get('hand', '')} V)"
            current_time = time.time()
            if current_time - self.last_violation_time["hand_signs"] > self.violation_cooldown: self.last_violation_time["hand_signs"] = current_time; self.save_violation(frame, "hand_signs", violation_details); print(f"VIOLATION: Hand Sign detected ({violation_details})")
        cv2.putText(frame, status_l, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        cv2.putText(frame, status_r, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


    # --- detect_passing_object REFINED with Motion Check ---
    def detect_passing_object(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask_cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask_cleaned = cv2.morphologyEx(fg_mask_cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        contours, _ = cv2.findContours(fg_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_potential_objects = []
        detected_this_frame_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter 1: Area
            if self.object_area_min < area < self.object_area_max:
                (x, y, w, h) = cv2.boundingRect(contour)
                # Filter 2: Aspect Ratio
                if h == 0: continue
                aspect_ratio = w / h
                if not (self.object_aspect_ratio_min < aspect_ratio < self.object_aspect_ratio_max): continue
                # Filter 3: Hand Overlap
                obj_bbox = (x, y, x + w, y + h)
                is_hand_overlap = False
                for hand_bbox in self.current_hand_bboxes:
                    xA=max(obj_bbox[0], hand_bbox[0]); yA=max(obj_bbox[1], hand_bbox[1]); xB=min(obj_bbox[2], hand_bbox[2]); yB=min(obj_bbox[3], hand_bbox[3])
                    interArea = max(0, xB - xA) * max(0, yB - yA); obj_area_bbox = w*h
                    if obj_area_bbox > 0 and (interArea / float(obj_area_bbox)) > self.hand_overlap_threshold: is_hand_overlap = True; break
                if is_hand_overlap: continue
                # Filter 4: Face Overlap
                is_face_overlap = False
                if self.current_face_bbox:
                    xA=max(obj_bbox[0], self.current_face_bbox[0]); yA=max(obj_bbox[1], self.current_face_bbox[1]); xB=min(obj_bbox[2], self.current_face_bbox[2]); yB=min(obj_bbox[3], self.current_face_bbox[3])
                    interArea = max(0, xB - xA) * max(0, yB - yA); obj_area_bbox = w*h
                    if obj_area_bbox > 0 and (interArea / float(obj_area_bbox)) > self.face_overlap_threshold: is_face_overlap = True
                if is_face_overlap: continue

                # Passed all filters - add to potentials for stability check
                center = (x + w // 2, y + h // 2)
                current_potential_objects.append({'center': center, 'area': area, 'rect': (x, y, w, h)})
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1) # Yellow
                detected_this_frame_count += 1

        # --- Stability & Motion Check ---
        stable_moving_objects_count = 0
        self.object_contour_buffer.append(current_potential_objects)
        consistent_objects_this_frame = [] # Objects that are stable

        if len(self.object_contour_buffer) == self.object_contour_buffer.maxlen:
            # Map current objects to objects in the first frame of the buffer for stability
            first_frame_objects = self.object_contour_buffer[0]
            # Use a simple matching: find closest past object within thresholds
            matched_indices = set() # Keep track of matched past objects
            for i, current_obj in enumerate(current_potential_objects):
                best_match_idx = -1
                min_dist = float('inf')
                for j, past_obj in enumerate(first_frame_objects):
                    if j in matched_indices: continue # Already matched

                    dist = math.dist(current_obj['center'], past_obj['center'])
                    area_ratio = current_obj['area'] / past_obj['area'] if past_obj['area'] > 0 else 0

                    # Check if it's a potential match based on stability criteria
                    if dist < 30 and 0.6 < area_ratio < 1.5: # Stability thresholds TUNE
                        if dist < min_dist: # Found a closer potential match
                            min_dist = dist
                            best_match_idx = j

                if best_match_idx != -1: # Found a stable match
                    matched_indices.add(best_match_idx)
                    consistent_objects_this_frame.append(current_obj) # Mark as stable

                    # *** NEW: Check Motion of this Stable Object ***
                    start_pos = first_frame_objects[best_match_idx]['center']
                    end_pos = current_obj['center']
                    stable_motion_dist = math.dist(start_pos, end_pos)

                    if stable_motion_dist >= self.object_min_stable_motion_px:
                        stable_moving_objects_count += 1
                        # Mark stable *and* moving objects differently (e.g., thicker red)
                        x, y, w, h = current_obj['rect']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2) # Red for stable+moving

        # --- Display Status and Trigger Violation ---
        status_text = f"Filtered Obj: {detected_this_frame_count}"; color = (0, 255, 0)

        # Trigger violation only if stable AND moving objects meet threshold
        if stable_moving_objects_count > 0: # Can adjust threshold (e.g., >= 1)
            # Add stable count to status for info
            status_text += f" (Stable: {len(consistent_objects_this_frame)}, Moving: {stable_moving_objects_count})"
            # Check if count meets violation threshold (could be just 1 stable moving object)
            if stable_moving_objects_count >= 1: # Trigger if at least one stable object moved sufficiently
                color = (0, 0, 255); status_text += " - OBJ?"
                current_time = time.time()
                if current_time - self.last_violation_time["passing_object"] > self.violation_cooldown:
                    self.last_violation_time["passing_object"] = current_time
                    self.save_violation(frame, "passing_object", {"count": stable_moving_objects_count})
                    print(f"VIOLATION: Suspicious Object detected (Stable & Moving count: {stable_moving_objects_count})")

        elif len(consistent_objects_this_frame) > 0 : # Some objects are stable but didn't move enough
             status_text += f" (Stable: {len(consistent_objects_this_frame)})"


        cv2.putText(frame, status_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


    # --- Main Loop and Drawing/Saving (No changes needed here) ---
    def run_detection_loop(self): # Same
        target_fps=15; frame_delay=1.0/target_fps; last_frame_time=time.time()
        while self.running:
            start_time=time.time(); ret, frame = self.cap.read()
            if not ret: print("Error: Failed frame grab."); self.running=False; break
            frame = cv2.flip(frame, 1); rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); rgb_frame.flags.writeable = False
            face_results = self.face_mesh.process(rgb_frame); hand_results = self.hands.process(rgb_frame); rgb_frame.flags.writeable = True
            self.detect_peeking(frame, face_results); self.detect_hand_signs(frame, hand_results); self.detect_passing_object(frame)
            self.draw_mediapipe_results(frame, face_results, hand_results)
            current_time=time.time(); fps=1.0/(current_time-last_frame_time) if (current_time-last_frame_time)>0 else 0; last_frame_time=current_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Exam Monitor - Press 'q' to Quit", frame)
            elapsed_time=time.time()-start_time; wait_time=max(1, int((frame_delay - elapsed_time)*1000))
            if cv2.waitKey(wait_time) & 0xFF == ord('q'): self.running = False; break
        self.stop()

    def draw_mediapipe_results(self, frame, face_results, hand_results): # Same
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks: self.mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks, connections=self.mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=self.drawing_spec_mesh)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks: self.mp_drawing.draw_landmarks(image=frame, landmark_list=hand_landmarks, connections=self.mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2), connection_drawing_spec=self.drawing_spec_hands)

    def save_violation(self, frame, violation_type, details={}): # Same
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]; filename_base=f"{violation_type}_{timestamp}"; filepath_img=os.path.join(self.output_folder, violation_type, f"{filename_base}.jpg")
        frame_copy=frame.copy(); text_y=self.frame_height-10; details_str=f"{violation_type.upper()} | {timestamp}"
        if violation_type=="peeking": details_str += f" | {details.get('reason','?')}"
        elif 'gesture' in details: details_str += f" | {details.get('hand','?')}:{details.get('gesture','?')}({details.get('count','?')})"
        elif 'count' in details: details_str += f" | ObjCount:{details['count']}"
        cv2.putText(frame_copy, details_str,(10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA); cv2.putText(frame_copy, details_str, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        try: cv2.imwrite(filepath_img, frame_copy); print(f"Saved violation: {filepath_img}")
        except Exception as e: print(f"ERROR: Failed to save image {filepath_img}: {e}")

def main(): # Same
    print("Initializing Malpractice Detection System..."); detector = MalpracticeDetector()
    if not detector.start(): print("Failed to start detector.")
    print("Detection system finished.")

if __name__ == "__main__": main()