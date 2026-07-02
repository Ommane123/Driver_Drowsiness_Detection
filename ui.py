import tkinter as tk
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import threading
import queue
import time
import sys
from detector import DrowsinessDetector
from audio import AudioAlarmController
import database

# Appearance and colors
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Theme colors mapping
STATE_COLORS = {
    "SAFE": "#2ecc71",       # Emerald Green
    "WARNING": "#f1c40f",    # Sunflower Yellow
    "DROWSY": "#e67e22",     # Carrot Orange
    "CRITICAL": "#e74c3c"    # Alizarin Red
}

class RealTimeGraph(ctk.CTkCanvas):
    """High-performance canvas-based graph to plot real-time EAR and MAR metrics."""
    def __init__(self, parent, width=380, height=180, max_points=100, **kwargs):
        super().__init__(parent, width=width, height=height, bg="#181825", highlightthickness=0, **kwargs)
        self.max_points = max_points
        self.ear_data = []
        self.mar_data = []
        self.width = width
        self.height = height
        self.ear_threshold = 0.20
        self.mar_threshold = 0.55
        
    def update_data(self, ear, mar, ear_thresh, mar_thresh):
        self.ear_data.append(ear)
        self.mar_data.append(mar)
        
        if len(self.ear_data) > self.max_points:
            self.ear_data.pop(0)
        if len(self.mar_data) > self.max_points:
            self.mar_data.pop(0)
            
        self.ear_threshold = ear_thresh
        self.mar_threshold = mar_thresh
        self.draw()
        
    def draw(self):
        self.delete("all")
        
        # Grid lines (y-axis)
        grid_color = "#313244"
        for i in range(1, 5):
            y = int(self.height * i / 5)
            self.create_line(0, y, self.width, y, fill=grid_color, dash=(2, 2))
            
        # Grid lines (x-axis)
        for i in range(1, 10):
            x = int(self.width * i / 10)
            self.create_line(x, 0, x, self.height, fill=grid_color, dash=(2, 2))
            
        # Scaling logic: map 0.0 - 0.70 to canvas coordinates
        max_val = 0.70
        def val_to_y(val):
            val = min(max_val, max(0.0, val))
            return int(self.height - (val / max_val) * self.height)

        # Plot EAR Threshold (Blue dotted line)
        y_ear_thresh = val_to_y(self.ear_threshold)
        self.create_line(0, y_ear_thresh, self.width, y_ear_thresh, fill="#3b82f6", dash=(4, 4), width=1.5)
        self.create_text(55, y_ear_thresh - 8, text=f"EAR Thresh: {self.ear_threshold:.2f}", fill="#3b82f6", font=("Inter", 8, "bold"))
        
        # Plot MAR Threshold (Yellow dotted line)
        y_mar_thresh = val_to_y(self.mar_threshold)
        self.create_line(0, y_mar_thresh, self.width, y_mar_thresh, fill="#f1c40f", dash=(4, 4), width=1.5)
        self.create_text(220, y_mar_thresh - 8, text=f"MAR Thresh: {self.mar_threshold:.2f}", fill="#f1c40f", font=("Inter", 8, "bold"))

        # Plot EAR Line (Blue)
        if len(self.ear_data) > 1:
            ear_points = []
            for idx, val in enumerate(self.ear_data):
                x = int(idx * (self.width / (self.max_points - 1)))
                y = val_to_y(val)
                ear_points.extend([x, y])
            try:
                self.create_line(ear_points, fill="#89b4fa", width=2, smooth=True)
            except Exception:
                self.create_line(ear_points, fill="#89b4fa", width=2) # Fallback if smooth fails

        # Plot MAR Line (Yellow)
        if len(self.mar_data) > 1:
            mar_points = []
            for idx, val in enumerate(self.mar_data):
                x = int(idx * (self.width / (self.max_points - 1)))
                y = val_to_y(val)
                mar_points.extend([x, y])
            try:
                self.create_line(mar_points, fill="#f9e2af", width=2, smooth=True)
            except Exception:
                self.create_line(mar_points, fill="#f9e2af", width=2)

import io

class DrowsinessApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ADAS - Driver Drowsiness & Distraction Monitor")
        self.geometry("1100x680")
        self.resizable(False, False)

        # Initialize session variables as None / inactive
        self.detector = None
        self.audio = None
        self.capture_thread = None
        self.stop_event = threading.Event()
        self.stop_event.set()  # Session is initially inactive
        self.session_photo_saved = False
        self.session_photo_id = None
        
        # Queues and thread synchronization
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Create Page Layouts
        self._build_home_ui()
        self._build_monitor_ui()
        
        # Load and display the last driver photo on Home Page
        self._load_and_display_last_driver_photo()

    def _build_home_ui(self):
        self.home_frame = ctk.CTkFrame(self, fg_color="#1e1e2e")
        self.home_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Inner container for home page to center content
        self.home_container = ctk.CTkFrame(
            self.home_frame, 
            fg_color="#181825", 
            corner_radius=15, 
            border_width=1, 
            border_color="#313244",
            width=700,
            height=520
        )
        self.home_container.place(relx=0.5, rely=0.5, anchor="center")

        # Title
        self.home_title = ctk.CTkLabel(
            self.home_container, 
            text="🚗 ADAS Safety Monitor", 
            font=("Outfit", 28, "bold"),
            text_color="#cdd6f4"
        )
        self.home_title.pack(pady=(35, 10))

        self.home_subtitle = ctk.CTkLabel(
            self.home_container, 
            text="Advanced Driver Assistance System for Drowsiness & Distraction Detection", 
            font=("Inter", 13, "italic"),
            text_color="#a6adc8"
        )
        self.home_subtitle.pack(pady=(0, 25))

        # Photo Frame Card
        self.photo_card = ctk.CTkFrame(self.home_container, fg_color="#252538", corner_radius=10, width=320, height=220)
        self.photo_card.pack(pady=10)
        self.photo_card.pack_propagate(False)

        self.photo_header = ctk.CTkLabel(
            self.photo_card,
            text="LAST DRIVER SESSION",
            font=("Outfit", 12, "bold"),
            text_color="#89b4fa"
        )
        self.photo_header.pack(pady=(8, 2))

        # Photo label inside the card
        self.driver_photo_label = ctk.CTkLabel(
            self.photo_card, 
            text="No previous driver photo.", 
            fg_color="#11111b",
            corner_radius=8,
            width=240,
            height=160
        )
        self.driver_photo_label.pack(pady=(5, 10))

        # Start button
        self.start_btn = ctk.CTkButton(
            self.home_container, 
            text="START MONITORING SESSION", 
            font=("Inter", 15, "bold"),
            fg_color="#2ecc71",
            hover_color="#27ae60",
            height=45,
            width=280,
            command=self._on_start_session
        )
        self.start_btn.pack(pady=(25, 10))

        # Distractions counter
        self.distraction_counter_lbl = ctk.CTkLabel(
            self.home_container,
            text="Distractions Detected: 0",
            font=("Outfit", 15, "bold"),
            text_color="#e74c3c"
        )
        self.distraction_counter_lbl.pack(pady=(0, 15))

        # Disclaimer
        self.disclaimer_lbl = ctk.CTkLabel(
            self.home_container, 
            text="Note: System requires camera access. Please calibrate eyes at the beginning of the session.", 
            font=("Inter", 10),
            text_color="#585b70"
        )
        self.disclaimer_lbl.pack(pady=(5, 15))

    def _build_monitor_ui(self):
        self.monitor_frame = ctk.CTkFrame(self, fg_color="transparent")
        # Grid Configuration for monitor_frame
        self.monitor_frame.grid_columnconfigure(0, weight=3) # Camera feed
        self.monitor_frame.grid_columnconfigure(1, weight=2) # Panel dashboard
        self.monitor_frame.grid_rowconfigure(0, weight=1)

        # Left Column Frame (Video Feed)
        self.left_frame = ctk.CTkFrame(self.monitor_frame, fg_color="#11111b")
        self.left_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        self.left_frame.grid_rowconfigure(1, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)

        # Title/Status Overlay Label
        self.title_label = ctk.CTkLabel(
            self.left_frame, 
            text="DRIVE SAFETY MONITOR", 
            font=("Outfit", 20, "bold"),
            text_color="#cdd6f4"
        )
        self.title_label.grid(row=0, column=0, pady=10, sticky="ew")

        # Camera Display Label
        self.video_display = ctk.CTkLabel(self.left_frame, text="", fg_color="#181825")
        self.video_display.grid(row=1, column=0, padx=15, pady=5, sticky="nsew")

        # Bottom Bar inside Camera Feed Frame
        self.camera_status = ctk.CTkLabel(
            self.left_frame, 
            text="Initializing webcam feed...", 
            font=("Inter", 12, "italic"),
            text_color="#a6adc8"
        )
        self.camera_status.grid(row=2, column=0, pady=10)

        # Right Column Frame (Dashboard panel)
        self.right_frame = ctk.CTkFrame(self.monitor_frame, fg_color="#1e1e2e")
        self.right_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        self.right_frame.grid_columnconfigure(0, weight=1)

        # --- PANEL SECTION 1: GLOBAL STATE ---
        self.state_panel = ctk.CTkFrame(self.right_frame, fg_color="#181825", corner_radius=10)
        self.state_panel.grid(row=0, column=0, padx=15, pady=10, sticky="ew")
        
        self.state_lbl = ctk.CTkLabel(
            self.state_panel, 
            text="SYSTEM STATE: SAFE", 
            font=("Outfit", 18, "bold"),
            text_color=STATE_COLORS["SAFE"]
        )
        self.state_lbl.pack(pady=8)

        self.risk_lbl = ctk.CTkLabel(
            self.state_panel, 
            text="TOTAL RISK SCORE: 0.0%", 
            font=("Inter", 13, "bold"),
            text_color="#cdd6f4"
        )
        self.risk_lbl.pack(pady=2)

        self.risk_bar = ctk.CTkProgressBar(self.state_panel, progress_color=STATE_COLORS["SAFE"])
        self.risk_bar.set(0.0)
        self.risk_bar.pack(fill="x", padx=30, pady=10)

        # Sub-metrics Grid (Eye, Yawn, Distraction progress bars)
        self.metrics_grid = ctk.CTkFrame(self.state_panel, fg_color="transparent")
        self.metrics_grid.pack(fill="x", padx=15, pady=5)
        self.metrics_grid.grid_columnconfigure((0, 1, 2), weight=1)

        # Eye Closure Metric Card
        self.eye_card = ctk.CTkFrame(self.metrics_grid, fg_color="#252538")
        self.eye_card.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.eye_lbl = ctk.CTkLabel(self.eye_card, text="Eye Closure\n0.0%", font=("Inter", 11))
        self.eye_lbl.pack(pady=4)
        self.eye_bar = ctk.CTkProgressBar(self.eye_card, height=6, progress_color="#3b82f6")
        self.eye_bar.set(0.0)
        self.eye_bar.pack(fill="x", padx=10, pady=5)

        # Yawn Metric Card
        self.yawn_card = ctk.CTkFrame(self.metrics_grid, fg_color="#252538")
        self.yawn_card.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.yawn_lbl = ctk.CTkLabel(self.yawn_card, text="Yawning\n0.0%", font=("Inter", 11))
        self.yawn_lbl.pack(pady=4)
        self.yawn_bar = ctk.CTkProgressBar(self.yawn_card, height=6, progress_color="#f1c40f")
        self.yawn_bar.set(0.0)
        self.yawn_bar.pack(fill="x", padx=10, pady=5)

        # Distraction Metric Card
        self.dist_card = ctk.CTkFrame(self.metrics_grid, fg_color="#252538")
        self.dist_card.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        self.dist_lbl = ctk.CTkLabel(self.dist_card, text="Distraction\n0.0%", font=("Inter", 11))
        self.dist_lbl.pack(pady=4)
        self.dist_bar = ctk.CTkProgressBar(self.dist_card, height=6, progress_color="#e74c3c")
        self.dist_bar.set(0.0)
        self.dist_bar.pack(fill="x", padx=10, pady=5)

        # --- PANEL SECTION 2: REAL-TIME PLOTS ---
        self.graph_panel = ctk.CTkFrame(self.right_frame, fg_color="#181825", corner_radius=10)
        self.graph_panel.grid(row=1, column=0, padx=15, pady=10, sticky="ew")
        
        self.graph_title = ctk.CTkLabel(
            self.graph_panel, 
            text="Metric Trends (EAR: Blue | MAR: Yellow)", 
            font=("Outfit", 12, "bold"),
            text_color="#a6adc8"
        )
        self.graph_title.pack(pady=4)

        self.graph = RealTimeGraph(self.graph_panel, width=420, height=140)
        self.graph.pack(padx=10, pady=5)

        # --- PANEL SECTION 3: SYSTEM SETTINGS ---
        self.control_panel = ctk.CTkFrame(self.right_frame, fg_color="#181825", corner_radius=10)
        self.control_panel.grid(row=2, column=0, padx=15, pady=10, sticky="ew")
        self.control_panel.grid_columnconfigure((0, 1), weight=1)

        # Calibrate and Alarm Test Buttons
        self.calib_btn = ctk.CTkButton(
            self.control_panel, 
            text="Calibrate Eyes", 
            font=("Inter", 12, "bold"),
            command=self._on_calibrate
        )
        self.calib_btn.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.test_sound_btn = ctk.CTkButton(
            self.control_panel, 
            text="Test Sound", 
            font=("Inter", 12, "bold"),
            fg_color="#313244",
            hover_color="#45475a",
            command=self._on_test_sound
        )
        self.test_sound_btn.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # Threshold slider adjustments
        self.sliders_frame = ctk.CTkFrame(self.control_panel, fg_color="transparent")
        self.sliders_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        self.sliders_frame.grid_columnconfigure(1, weight=1)

        # EAR Slider
        ctk.CTkLabel(self.sliders_frame, text="EAR Threshold:", font=("Inter", 11)).grid(row=0, column=0, padx=5, sticky="w")
        self.ear_slider = ctk.CTkSlider(self.sliders_frame, from_=0.10, to=0.35, command=self._on_ear_slider)
        self.ear_slider.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.ear_val_lbl = ctk.CTkLabel(self.sliders_frame, text="0.21", font=("Inter", 11, "bold"), width=35)
        self.ear_val_lbl.grid(row=0, column=2, padx=5)

        # MAR Slider
        ctk.CTkLabel(self.sliders_frame, text="MAR Threshold:", font=("Inter", 11)).grid(row=1, column=0, padx=5, sticky="w")
        self.mar_slider = ctk.CTkSlider(self.sliders_frame, from_=0.30, to=0.80, command=self._on_mar_slider)
        self.mar_slider.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.mar_val_lbl = ctk.CTkLabel(self.sliders_frame, text="0.55", font=("Inter", 11, "bold"), width=35)
        self.mar_val_lbl.grid(row=1, column=2, padx=5)

        # Exit Session Button (Wide RED button at the bottom of the settings panel)
        self.exit_session_btn = ctk.CTkButton(
            self.control_panel,
            text="Exit Session",
            font=("Inter", 12, "bold"),
            fg_color="#e74c3c",
            hover_color="#c0392b",
            command=self._on_exit_session
        )
        self.exit_session_btn.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

    def _on_start_session(self):
        # 1. Switch layouts
        self.home_frame.grid_forget()
        self.monitor_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")

        # 2. Reset synchronization primitives
        self.stop_event = threading.Event()
        self.stop_event.clear()
        self.session_photo_saved = False
        self.session_photo_id = None

        # 3. Instantiate detector and audio controller
        self.camera_status.configure(text="Initializing detector & camera...", text_color="#a6adc8")
        self.detector = DrowsinessDetector()
        self.audio = AudioAlarmController()

        # 4. Synchronize GUI sliders with detector configuration
        self.ear_slider.set(self.detector.ear_threshold)
        self.ear_val_lbl.configure(text=f"{self.detector.ear_threshold:.2f}")
        self.mar_slider.set(self.detector.mar_threshold)
        self.mar_val_lbl.configure(text=f"{self.detector.mar_threshold:.2f}")

        # 5. Start camera thread
        self.capture_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.capture_thread.start()

        # 6. Start UI poll loops
        self._poll_frame()

    def _on_exit_session(self):
        # 1. Signal camera loop to stop
        self.stop_event.set()

        # 2. Wait for camera thread to exit cleanly
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.5)
            self.capture_thread = None

        # 3. Save distraction count to database before shutting down audio
        play_count = self.audio.play_count if self.audio else 0
        if self.session_photo_id is not None:
            try:
                database.update_driver_distractions(self.session_photo_id, play_count)
            except Exception as e:
                print(f"[UI ERROR] Failed to update driver distractions count: {e}")

        # 4. Shutdown detector and audio alerts
        if self.detector:
            try:
                self.detector.close()
            except Exception:
                pass
            self.detector = None

        if self.audio:
            try:
                self.audio.shutdown()
            except Exception:
                pass
            self.audio = None

        # 5. Switch layouts back to Home Page
        self.monitor_frame.grid_forget()
        self.home_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")

        # 5. Reload the last driver photo (so the captured photo shows up)
        self._load_and_display_last_driver_photo()

    def _on_test_sound(self):
        if self.audio:
            self.audio.test_alert()

    def _on_ear_slider(self, val):
        if self.detector:
            val = round(float(val), 2)
            self.detector.ear_threshold = val
            self.ear_val_lbl.configure(text=f"{val:.2f}")

    def _on_mar_slider(self, val):
        if self.detector:
            val = round(float(val), 2)
            self.detector.mar_threshold = val
            self.mar_val_lbl.configure(text=f"{val:.2f}")

    def _on_calibrate(self):
        if self.detector:
            self.detector.start_calibration()
            self.calib_btn.configure(text="Calibrating...", state="disabled")

    def _load_and_display_last_driver_photo(self):
        try:
            photo_bytes, distractions = database.fetch_last_driver_photo_and_distractions()
            if photo_bytes:
                pil_img = Image.open(io.BytesIO(photo_bytes))
                self.last_driver_image = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(240, 160))
                self.driver_photo_label.configure(image=self.last_driver_image, text="")
            else:
                self.driver_photo_label.configure(image=None, text="No previous driver photo.")
            
            # Update the distraction counter label
            if hasattr(self, 'distraction_counter_lbl'):
                self.distraction_counter_lbl.configure(text=f"Distractions Detected: {distractions}")
        except Exception as e:
            print(f"[UI ERROR] Failed to load last driver photo and distractions: {e}")
            self.driver_photo_label.configure(image=None, text="No previous driver photo.")
            if hasattr(self, 'distraction_counter_lbl'):
                self.distraction_counter_lbl.configure(text="Distractions Detected: 0")

    def _camera_loop(self):
        """Worker thread loop. Captures frames from webcam and analyzes landmarks."""
        cap = None
        # Attempt to open indices 0, then 1, then 2
        for camera_idx in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_idx)
            if cap.isOpened():
                print(f"[CAMERA] Opened camera source index: {camera_idx}")
                break
            cap.release()
            cap = None
            
        if cap is None:
            self.frame_queue.put({"error": "Failed to connect to webcam."})
            return

        # Read frames continuously
        fps_time = time.time()
        frame_count = 0
        current_fps = 30.0

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            frame_count += 1
            if frame_count >= 15:
                now = time.time()
                current_fps = frame_count / (now - fps_time + 1e-6)
                fps_time = now
                frame_count = 0

            # Analyze frame metrics on the raw unflipped frame first (ensures accurate head pose coordinates)
            results = self.detector.process_frame(frame) if self.detector else {"face_detected": False}
            results["fps"] = current_fps
            
            # Flip frame horizontally (including its drawn overlays) for a natural mirror view
            frame = cv2.flip(frame, 1)
            
            # Capture driver's photo once when a face is first successfully detected in this session
            if results.get("face_detected") and not self.session_photo_saved:
                self.session_photo_saved = True
                success, encoded_image = cv2.imencode('.jpg', frame)
                if success:
                    photo_bytes = encoded_image.tobytes()
                    try:
                        self.session_photo_id = database.save_driver_photo(photo_bytes)
                        # Schedule updating the home screen last driver photo frame
                        self.after(0, self._load_and_display_last_driver_photo)
                    except Exception as e:
                        print(f"[CAMERA ERROR] Failed to save driver photo to database: {e}")
            
            # Prepare image for Tkinter display
            # OpenCV is BGR, PIL needs RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            # Resize image to fit panel (approx 580x420 px)
            pil_img = pil_img.resize((580, 420), Image.Resampling.LANCZOS)
            
            results["img_pil"] = pil_img

            # Push results into Queue, discard old frame if UI is lagging
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(results)

        cap.release()

    def _poll_frame(self):
        """Main thread callback. Polls the queue for analyzed frames and updates widgets."""
        if self.stop_event.is_set():
            return  # Stop polling when session is inactive!

        try:
            while True:
                results = self.frame_queue.get_nowait()
                
                # Check for capture errors
                if "error" in results:
                    self.camera_status.configure(text=results["error"], text_color="#f38ba8")
                    self.video_display.configure(text="Webcam Error. Verify connections.", text_color="#f38ba8")
                    return
                
                # Update webcam label image using CTkImage to support scaling and prevent HighDPI warnings
                ctk_img = ctk.CTkImage(light_image=results["img_pil"], dark_image=results["img_pil"], size=(580, 420))
                self.video_display.configure(image=ctk_img, text="")
                self.video_display.image = ctk_img  # Keep reference

                # Update camera bar status
                face_status = "Face Active" if results["face_detected"] else "FACE LOST"
                face_color = "#a6e3a1" if results["face_detected"] else "#f38ba8"
                self.camera_status.configure(
                    text=f"{face_status} | FPS: {results['fps']:.1f} | EAR: {results['ear']:.2f} | MAR: {results['mar']:.2f}",
                    text_color=face_color
                )

                # Update settings calibration status
                if self.detector:
                    if results["calibrating"]:
                        pct = int(results["calibration_progress"] * 100)
                        self.calib_btn.configure(text=f"Calibrating ({pct}%)")
                    else:
                        self.calib_btn.configure(text="Calibrate Eyes", state="normal")
                        # Synchronize calibration updates back to GUI sliders
                        self.ear_slider.set(self.detector.ear_threshold)
                        self.ear_val_lbl.configure(text=f"{self.detector.ear_threshold:.2f}")

                # Update Audio State
                if self.audio:
                    self.audio.set_state(results["state"])

                # Update UI Dashboard Panel Info
                self._update_dashboard(results)
                
                # Update line graph canvas
                if self.detector:
                    self.graph.update_data(
                        ear=results["ear"], 
                        mar=results["mar"], 
                        ear_thresh=self.detector.ear_threshold, 
                        mar_thresh=self.detector.mar_threshold
                    )
                
        except queue.Empty:
            pass
            
        # Poll again in 15 milliseconds (approx 60fps refresh rate)
        self.after(15, self._poll_frame)

    def _update_dashboard(self, results):
        state = results["state"]
        risk = results["risk_score"]
        
        # Color transition for status label
        self.state_lbl.configure(
            text=f"SYSTEM STATE: {state}",
            text_color=STATE_COLORS[state]
        )
        self.risk_lbl.configure(text=f"TOTAL RISK SCORE: {risk:.1f}%")
        
        # Risk progress bar
        self.risk_bar.set(risk / 100.0)
        self.risk_bar.configure(progress_color=STATE_COLORS[state])

        # Eye Closure details
        eye_pct = results["eye_score"] * 100.0
        self.eye_lbl.configure(text=f"Eye Closure\n{eye_pct:.1f}%")
        self.eye_bar.set(results["eye_score"])

        # Yawn details
        yawn_pct = results["yawn_score"] * 100.0
        self.yawn_lbl.configure(text=f"Yawning\n{yawn_pct:.1f}%")
        self.yawn_bar.set(results["yawn_score"])

        # Distraction details
        dist_pct = results["distraction_score"] * 100.0
        self.dist_lbl.configure(
            text=f"Distraction\n{dist_pct:.1f}%",
            text_color="#f38ba8" if dist_pct > 30 else "#cdd6f4"
        )
        self.dist_bar.set(results["distraction_score"])

    def on_closing(self):
        """Handles proper shutdown of threads and audio mixers on window exit."""
        print("[UI] Window closing, clean up resources...")
        self.stop_event.set()
        
        # Wait for camera thread to exit first (prevents background thread accessing closed detector)
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.5)
            
        # Shutdown alerts and detector
        if self.audio:
            try:
                self.audio.shutdown()
            except Exception:
                pass
        if self.detector:
            try:
                self.detector.close()
            except Exception:
                pass
            
        self.destroy()
        sys.exit(0)

if __name__ == "__main__":
    app = DrowsinessApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
