import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
from detector import DrowsinessDetector

# Page settings
st.set_page_config(
    page_title="ADAS Driver Drowsiness Portal", 
    page_icon="🚗", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #11111b;
    }
    h1 {
        color: #cdd6f4 !important;
        font-family: 'Outfit', sans-serif;
    }
    h2, h3 {
        color: #bac2de !important;
        font-family: 'Outfit', sans-serif;
    }
    .metric-card {
        background-color: #181825;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #313244;
    }
</style>
""", unsafe_allow_html=True)

class DrowsinessVideoProcessor(VideoProcessorBase):
    """Processes WebRTC video frames and draws landmarks and head-pose axes on the client side."""
    def __init__(self):
        # Instantiate detector per user connection
        self.detector = DrowsinessDetector()
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # Run detection
        results = self.detector.process_frame(img)
        
        state = results["state"]
        risk = results["risk_score"]
        
        # Color mapping (BGR)
        color_map = {
            "SAFE": (113, 204, 46),      # Green
            "WARNING": (15, 196, 241),   # Yellow
            "DROWSY": (34, 126, 230),    # Orange
            "CRITICAL": (60, 76, 231)    # Red
        }
        color = color_map.get(state, (255, 255, 255))
        
        # Overlay System Status directly on the video frames
        cv2.putText(
            img, 
            f"STATE: {state}", 
            (20, 45), 
            cv2.FONT_HERSHEY_DUPLEX, 
            1.0, 
            color, 
            2, 
            cv2.LINE_AA
        )
        cv2.putText(
            img, 
            f"RISK: {risk:.1f}%", 
            (20, 85), 
            cv2.FONT_HERSHEY_DUPLEX, 
            1.0, 
            color, 
            2, 
            cv2.LINE_AA
        )
        
        # Paint sub-metrics
        cv2.putText(
            img, 
            f"Eyes: {results['eye_score']*100:.0f}%  Yawn: {results['yawn_score']*100:.0f}%  Pose: {results['distraction_score']*100:.0f}%", 
            (20, h - 25), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (205, 214, 244), 
            1, 
            cv2.LINE_AA
        )

        # Draw a heavy safety outline overlay if alert state is elevated
        if state in ["DROWSY", "CRITICAL"]:
            # Pulsing color border thickness
            cv2.rectangle(img, (0, 0), (w, h), color, 8)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    # Sidebar
    st.sidebar.title("ℹ️ ADAS Info & Specs")
    st.sidebar.info("""
    **Weighted Scoring Matrix:**
    * **Eye Closure Score**: 50% weight (Max at 4.0s)
    * **Yawning Score**: 20% weight (Max at 3.0s)
    * **Distraction Score**: 30% weight (Max at 3.0s)
    
    **Risk Severity Thresholds:**
    * **0 - 30%**: SAFE (Green)
    * **31 - 60%**: WARNING (Yellow)
    * **61 - 80%**: DROWSY (Orange)
    * **81 - 100%**: CRITICAL (Red)
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### local SQLite Logging")
    st.sidebar.write("Desktop versions save events locally. Web servers run ephemerally per user session.")

    # Main content layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.title("🚗 ADAS Driver Safety Portal")
        st.write("Start the WebRTC Webcam stream below to test the drowsiness detection engine.")

        # WebRTC stream configuration
        # Use standard free Google STUN servers for WebRTC NAT traversal
        rtc_config = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        webrtc_streamer(
            key="drowsiness-detection",
            video_processor_factory=DrowsinessVideoProcessor,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

    with col2:
        st.markdown("### 📊 Metric Descriptions")
        
        st.markdown("""
        <div class="metric-card">
            <h4>👁️ Eye Aspect Ratio (EAR)</h4>
            <p>Monitors vertical-to-horizontal eye openings. Continuous low EAR signifies blinking or eye-closure.</p>
        </div>
        <br>
        <div class="metric-card">
            <h4>😮 Mouth Aspect Ratio (MAR)</h4>
            <p>Calculates lip coordinates. High MAR indicates gaping or yawning events.</p>
        </div>
        <br>
        <div class="metric-card">
            <h4>📐 Head Pose (Yaw/Pitch)</h4>
            <p>Measures head rotation angle in 3D. Used to identify when the driver shifts attention away from the road.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
