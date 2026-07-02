import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
from collections import deque
import time
import os
import urllib.request
import database

# Path to local model file
MODEL_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task"))

def download_model_if_needed():
    """Programmatically downloads the official Google Face Landmarker model file if missing."""
    if not os.path.exists(MODEL_PATH):
        print(f"[DETECTOR] Downloading face_landmarker.task to: {MODEL_PATH}...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        try:
            # Download model file
            urllib.request.urlretrieve(url, MODEL_PATH)
            print("[DETECTOR] Model downloaded successfully.")
        except Exception as e:
            print(f"[DETECTOR ERROR] Failed to download Face Landmarker model file: {e}")
            raise

class DrowsinessDetector:
    def __init__(self, ear_threshold=0.21, mar_threshold=0.55, yaw_threshold=22.0, pitch_threshold=16.0):
        # Ensure model is downloaded
        download_model_if_needed()
        
        # Configure MediaPipe Tasks Face Landmarker
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # Thresholds (can be calibrated dynamically)
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        
        # Temporal buffers (rolling windows)
        self.ear_history = deque(maxlen=45)       # ~1.5s at 30fps
        self.mar_history = deque(maxlen=30)       # ~1.0s at 30fps
        self.yaw_history = deque(maxlen=45)       # ~1.5s at 30fps
        self.pitch_history = deque(maxlen=45)     # ~1.5s at 30fps
        
        # Cumulative warning times (in seconds)
        self.eyes_closed_time = 0.0
        self.yawning_time = 0.0
        self.distracted_time = 0.0
        
        # Timing trackers
        self.last_frame_time = time.time()
        
        # Dynamic Calibration state
        self.is_calibrating = False
        self.calibration_frames = []
        self.calibration_limit = 90  # 3 seconds of calibration data at 30fps
        
        # Weighted Scoring Risk
        self.risk_score = 0.0
        self.current_state = "SAFE"
        
        # State tracking for DB logging
        self.state_entry_time = time.time()
        self.max_risk_in_current_state = 0.0
        self.state_ear_values = []
        self.state_mar_values = []
        self.state_yaw_values = []
        self.state_pitch_values = []
        
        # Head pose 3D model points (centered generic head coordinates)
        self.model_3d_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip (Landmark 1)
            (0.0, 330.0, -65.0),         # Chin (Landmark 152) - Y is positive (down)
            (225.0, -170.0, -135.0),     # Left eye outer corner (Landmark 263) - Y is negative (up)
            (-225.0, -170.0, -135.0),    # Right eye outer corner (Landmark 33) - Y is negative (up)
            (150.0, 150.0, -125.0),      # Left mouth corner (Landmark 287) - Y is positive (down)
            (-150.0, 150.0, -125.0)      # Right mouth corner (Landmark 57) - Y is positive (down)
        ], dtype=np.float32)

    def _euclidean_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _calculate_ear(self, landmarks, w, h):
        # Left eye landmarks: Inner (362), Outer (263), Top (385, 386, 387), Bottom (380, 374, 373)
        p_362 = (landmarks[362].x * w, landmarks[362].y * h)
        p_263 = (landmarks[263].x * w, landmarks[263].y * h)
        p_385 = (landmarks[385].x * w, landmarks[385].y * h)
        p_380 = (landmarks[380].x * w, landmarks[380].y * h)
        p_386 = (landmarks[386].x * w, landmarks[386].y * h)
        p_374 = (landmarks[374].x * w, landmarks[374].y * h)
        p_387 = (landmarks[387].x * w, landmarks[387].y * h)
        p_373 = (landmarks[373].x * w, landmarks[373].y * h)

        ear_left = (self._euclidean_distance(p_385, p_380) + 
                    self._euclidean_distance(p_386, p_374) + 
                    self._euclidean_distance(p_387, p_373)) / (3.0 * self._euclidean_distance(p_362, p_263) + 1e-6)

        # Right eye landmarks: Inner (133), Outer (33), Top (160, 159, 158), Bottom (144, 145, 153)
        p_133 = (landmarks[133].x * w, landmarks[133].y * h)
        p_33 = (landmarks[33].x * w, landmarks[33].y * h)
        p_160 = (landmarks[160].x * w, landmarks[160].y * h)
        p_144 = (landmarks[144].x * w, landmarks[144].y * h)
        p_159 = (landmarks[159].x * w, landmarks[159].y * h)
        p_145 = (landmarks[145].x * w, landmarks[145].y * h)
        p_158 = (landmarks[158].x * w, landmarks[158].y * h)
        p_153 = (landmarks[153].x * w, landmarks[153].y * h)

        ear_right = (self._euclidean_distance(p_160, p_144) + 
                     self._euclidean_distance(p_159, p_145) + 
                     self._euclidean_distance(p_158, p_153)) / (3.0 * self._euclidean_distance(p_133, p_33) + 1e-6)

        # Average EAR
        return (ear_left + ear_right) / 2.0

    def _calculate_mar(self, landmarks, w, h):
        # Mouth inner landmarks: Outer corners (78, 308), Top (82, 13, 312), Bottom (87, 14, 317)
        p_78 = (landmarks[78].x * w, landmarks[78].y * h)
        p_308 = (landmarks[308].x * w, landmarks[308].y * h)
        p_82 = (landmarks[82].x * w, landmarks[82].y * h)
        p_87 = (landmarks[87].x * w, landmarks[87].y * h)
        p_13 = (landmarks[13].x * w, landmarks[13].y * h)
        p_14 = (landmarks[14].x * w, landmarks[14].y * h)
        p_312 = (landmarks[312].x * w, landmarks[312].y * h)
        p_317 = (landmarks[317].x * w, landmarks[317].y * h)

        mar = (self._euclidean_distance(p_82, p_87) + 
               self._euclidean_distance(p_13, p_14) + 
               self._euclidean_distance(p_312, p_317)) / (3.0 * self._euclidean_distance(p_78, p_308) + 1e-6)
        return mar

    def _estimate_head_pose(self, landmarks, w, h):
        # Extract 2D coordinates mapping the 3D model landmarks
        # Landmarks: Nose (1), Chin (152), Left outer eye (263), Right outer eye (33), Left mouth corner (287), Right mouth corner (57)
        image_points = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),
            (landmarks[152].x * w, landmarks[152].y * h),
            (landmarks[263].x * w, landmarks[263].y * h),
            (landmarks[33].x * w, landmarks[33].y * h),
            (landmarks[287].x * w, landmarks[287].y * h),
            (landmarks[57].x * w, landmarks[57].y * h)
        ], dtype=np.float32)

        # Camera calibration approximation
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Assuming no distortion
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(self.model_3d_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return 0.0, 0.0

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        # Calculate euler angles
        sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(rmat[2, 1], rmat[2, 2])
            yaw = math.atan2(-rmat[2, 0], sy)
            # roll = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
            yaw = math.atan2(-rmat[2, 0], sy)

        # Convert to degrees
        return np.degrees(yaw), np.degrees(pitch)

    def start_calibration(self):
        """Triggers the start of the dynamic calibration sequence."""
        self.is_calibrating = True
        self.calibration_frames = []
        print("[DETECTOR] Calibration started. Please look straight at the camera and keep eyes open.")

    def process_frame(self, frame):
        current_time = time.time()
        time_elapsed = current_time - self.last_frame_time
        self.last_frame_time = current_time

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert RGB image data to MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.landmarker.detect(mp_image)

        face_detected = False
        ear, mar, yaw, pitch = 0.0, 0.0, 0.0, 0.0
        landmarks = None
        
        # Color palette for overlays based on current risk state
        color_map = {
            "SAFE": (46, 204, 113),      # Green
            "WARNING": (241, 196, 15),   # Yellow/Amber
            "DROWSY": (230, 126, 34),    # Orange
            "CRITICAL": (231, 76, 60)    # Red
        }
        draw_color = color_map.get(self.current_state, (255, 255, 255))
        # Convert RGB to BGR for OpenCV drawing
        cv_draw_color = (int(draw_color[2]), int(draw_color[1]), int(draw_color[0]))

        if detection_result.face_landmarks:
            face_detected = True
            landmarks = detection_result.face_landmarks[0]

            # Calculate instant metrics
            ear = self._calculate_ear(landmarks, w, h)
            mar = self._calculate_mar(landmarks, w, h)
            yaw, pitch = self._estimate_head_pose(landmarks, w, h)

            # Perform dynamic calibration
            if self.is_calibrating:
                self.calibration_frames.append(ear)
                if len(self.calibration_frames) >= self.calibration_limit:
                    avg_calib_ear = np.mean(self.calibration_frames)
                    # Set EAR threshold to 70% of the baseline open-eye state
                    self.ear_threshold = round(avg_calib_ear * 0.70, 2)
                    self.is_calibrating = False
                    print(f"[DETECTOR] Calibration complete! Base EAR: {avg_calib_ear:.3f}. Threshold set to {self.ear_threshold:.2f}")

            # Append to temporal histories
            self.ear_history.append(ear)
            self.mar_history.append(mar)
            self.yaw_history.append(yaw)
            self.pitch_history.append(pitch)

            # --- DRAW VISUAL OVERLAYS ---
            # 1. Draw Eye Landmark Dots
            left_eye_indices = [362, 263, 385, 380, 386, 374, 387, 373]
            right_eye_indices = [133, 33, 160, 144, 159, 145, 158, 153]
            for idx in left_eye_indices + right_eye_indices:
                pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
                cv2.circle(frame, pt, 2, cv_draw_color, -1)
                
            # 2. Draw Mouth Landmark Dots
            mouth_indices = [78, 308, 82, 87, 13, 14, 312, 317]
            for idx in mouth_indices:
                pt = (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
                cv2.circle(frame, pt, 2, (0, 255, 255), -1)  # Yellow for mouth

            # 3. Draw 3D Head Pose Axes
            focal_length = w
            center = (w / 2.0, h / 2.0)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float32)
            dist_coeffs = np.zeros((4, 1), dtype=np.float32)

            image_points = np.array([
                (landmarks[1].x * w, landmarks[1].y * h),
                (landmarks[152].x * w, landmarks[152].y * h),
                (landmarks[263].x * w, landmarks[263].y * h),
                (landmarks[33].x * w, landmarks[33].y * h),
                (landmarks[287].x * w, landmarks[287].y * h),
                (landmarks[57].x * w, landmarks[57].y * h)
            ], dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(self.model_3d_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if success:
                axis_points = np.array([
                    (80.0, 0.0, 0.0),    # X axis (Right - Red)
                    (0.0, 80.0, 0.0),    # Y axis (Up - Green)
                    (0.0, 0.0, 80.0)     # Z axis (Forward/Out - Blue)
                ], dtype=np.float32)
                
                nose_tip_2d = (int(image_points[0][0]), int(image_points[0][1]))
                imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
                
                cv2.line(frame, nose_tip_2d, (int(imgpts[0][0][0]), int(imgpts[0][0][1])), (0, 0, 255), 2)  # X-Axis (Red)
                cv2.line(frame, nose_tip_2d, (int(imgpts[1][0][0]), int(imgpts[1][0][1])), (0, 255, 0), 2)  # Y-Axis (Green)
                cv2.line(frame, nose_tip_2d, (int(imgpts[2][0][0]), int(imgpts[2][0][1])), (255, 0, 0), 2)  # Z-Axis (Blue)
        else:
            # Face tracking lost: decay metrics slowly towards safe baseline
            if len(self.ear_history) > 0:
                self.ear_history.append(self.ear_threshold + 0.05)
            if len(self.mar_history) > 0:
                self.mar_history.append(0.1)
            if len(self.yaw_history) > 0:
                self.yaw_history.append(0.0)
            if len(self.pitch_history) > 0:
                self.pitch_history.append(0.0)

        # Compute rolling window averages
        avg_ear = np.mean(self.ear_history) if self.ear_history else 0.25
        avg_mar = np.mean(self.mar_history) if self.mar_history else 0.15
        avg_yaw = np.mean(self.yaw_history) if self.yaw_history else 0.0
        avg_pitch = np.mean(self.pitch_history) if self.pitch_history else 0.0

        # --- 1. Evaluate Eyes (Eye Closure Score) ---
        if avg_ear < self.ear_threshold:
            self.eyes_closed_time += time_elapsed
        else:
            self.eyes_closed_time = max(0.0, self.eyes_closed_time - time_elapsed * 2.0)
        
        # Eye Closure Score (maxes at 4.0s)
        eye_score = min(1.0, self.eyes_closed_time / 4.0)

        # --- 2. Evaluate Yawning (Yawn Score) ---
        if avg_mar > self.mar_threshold:
            self.yawning_time += time_elapsed
        else:
            self.yawning_time = max(0.0, self.yawning_time - time_elapsed * 2.0)
            
        # Yawn Score (maxes at 3.0s)
        yawn_score = min(1.0, self.yawning_time / 3.0)

        # --- 3. Evaluate Distraction (Head Pose Score) ---
        is_distracted = abs(avg_yaw) > self.yaw_threshold or abs(avg_pitch) > self.pitch_threshold
        if is_distracted:
            self.distracted_time += time_elapsed
        else:
            self.distracted_time = max(0.0, self.distracted_time - time_elapsed * 2.0)
            
        # Distraction Score (maxes at 3.0s)
        distraction_score = min(1.0, self.distracted_time / 3.0)

        # --- Calculate Weighted ADAS Risk Score ---
        # Weights: Eye = 0.5, Yawn = 0.2, Distraction = 0.3
        self.risk_score = (eye_score * 0.5 + yawn_score * 0.2 + distraction_score * 0.3) * 100.0

        # --- Determine Severity State ---
        # 0-30 SAFE, 31-60 WARNING, 61-80 DROWSY, 81+ CRITICAL
        new_state = "SAFE"
        if self.risk_score > 80.0:
            new_state = "CRITICAL"
        elif self.risk_score > 60.0:
            new_state = "DROWSY"
        elif self.risk_score > 30.0:
            new_state = "WARNING"

        # Check for State Transitions to log events to SQLite database
        if new_state != self.current_state:
            # We transition out of a state -> log the previous state event if it was not SAFE
            if self.current_state != "SAFE":
                duration = time.time() - self.state_entry_time
                if duration >= 0.5:
                    database.log_event(
                        event_type=self.current_state,
                        duration=round(duration, 2),
                        max_risk_score=round(self.max_risk_in_current_state, 1),
                        avg_ear=round(np.mean(self.state_ear_values), 3) if self.state_ear_values else 0.0,
                        avg_mar=round(np.mean(self.state_mar_values), 3) if self.state_mar_values else 0.0,
                        avg_head_yaw=round(np.mean(self.state_yaw_values), 1) if self.state_yaw_values else 0.0,
                        avg_head_pitch=round(np.mean(self.state_pitch_values), 1) if self.state_pitch_values else 0.0
                    )
            
            # Reset transition stats
            self.current_state = new_state
            self.state_entry_time = time.time()
            self.max_risk_in_current_state = self.risk_score
            self.state_ear_values = [ear] if face_detected else []
            self.state_mar_values = [mar] if face_detected else []
            self.state_yaw_values = [yaw] if face_detected else []
            self.state_pitch_values = [pitch] if face_detected else []
        else:
            self.max_risk_in_current_state = max(self.max_risk_in_current_state, self.risk_score)
            if face_detected:
                self.state_ear_values.append(ear)
                self.state_mar_values.append(mar)
                self.state_yaw_values.append(yaw)
                self.state_pitch_values.append(pitch)

        # Assemble package of values for UI consumption
        return {
            "face_detected": face_detected,
            "landmarks": landmarks,
            "ear": ear,
            "avg_ear": avg_ear,
            "mar": mar,
            "avg_mar": avg_mar,
            "yaw": yaw,
            "pitch": pitch,
            "eye_score": eye_score,
            "yawn_score": yawn_score,
            "distraction_score": distraction_score,
            "risk_score": self.risk_score,
            "state": self.current_state,
            "calibrating": self.is_calibrating,
            "calibration_progress": len(self.calibration_frames) / self.calibration_limit if self.is_calibrating else 0.0
        }
        
    def close(self):
        """Cleanup detector resources."""
        self.landmarker.close()
        # Log final pending state if it was warning/drowsy/critical
        if self.current_state != "SAFE":
            duration = time.time() - self.state_entry_time
            if duration >= 0.5:
                try:
                    database.log_event(
                        event_type=self.current_state,
                        duration=round(duration, 2),
                        max_risk_score=round(self.max_risk_in_current_state, 1),
                        avg_ear=round(np.mean(self.state_ear_values), 3) if self.state_ear_values else 0.0,
                        avg_mar=round(np.mean(self.state_mar_values), 3) if self.state_mar_values else 0.0,
                        avg_head_yaw=round(np.mean(self.state_yaw_values), 1) if self.state_yaw_values else 0.0,
                        avg_head_pitch=round(np.mean(self.state_pitch_values), 1) if self.state_pitch_values else 0.0
                    )
                except Exception:
                    pass
