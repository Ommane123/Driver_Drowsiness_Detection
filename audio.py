import os
import threading
import time
import winsound

class AudioAlarmController:
    def __init__(self):
        self.sound_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "beep.wav"))
        self.enabled = False
        self.current_state = "SAFE"
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.play_count = 0  # Total plays during session
        
        if os.path.exists(self.sound_path):
            self.enabled = True
            print(f"[AUDIO] Audio system initialized successfully with: {self.sound_path}")
        else:
            print(f"[AUDIO WARNING] beep.wav not found at {self.sound_path}. Audio alerts will be disabled.")

    def set_state(self, state):
        """Sets the severity state and updates alarm playback.
        
        States:
        - SAFE: No sound.
        - WARNING: Intermittent beep (every 2.5 seconds).
        - DROWSY: Continuous beep looping.
        - CRITICAL: Continuous beep looping.
        """
        if not self.enabled:
            return
            
        if self.current_state == state:
            return
            
        self.current_state = state
        self._apply_state()

    def _apply_state(self):
        # Stop any playing audio immediately
        winsound.PlaySound(None, winsound.SND_PURGE)
        
        # Signal any running warning loop thread to stop
        self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=0.2)
        self.stop_event.clear()

        if self.current_state == "SAFE":
            pass
        elif self.current_state == "WARNING":
            # Start warning beep thread (plays sound every 2.5 seconds)
            self.worker_thread = threading.Thread(target=self._warning_loop, daemon=True)
            self.worker_thread.start()
        elif self.current_state in ["DROWSY", "CRITICAL"]:
            # Play sound continuously in loop asynchronously
            winsound.PlaySound(self.sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
            self.play_count += 1

    def _warning_loop(self):
        while not self.stop_event.is_set():
            winsound.PlaySound(self.sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            self.play_count += 1
            # Sleep for 2.5 seconds, check stop_event every 100ms for fast response
            for _ in range(25):
                if self.stop_event.is_set():
                    break
                time.sleep(0.1)

    def test_alert(self):
        """Plays a single test beep."""
        if self.enabled:
            winsound.PlaySound(self.sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)

    def shutdown(self):
        """Clean up audio assets."""
        self.stop_event.set()
        winsound.PlaySound(None, winsound.SND_PURGE)
        print("[AUDIO] Audio system shut down.")
