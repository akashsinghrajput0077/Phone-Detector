import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from ultralytics import YOLO
import av
import time
import threading

st.set_page_config(
    page_title="No Phone Zone",
    page_icon="🚫",
    layout="centered"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    * { font-family: 'Space Grotesk', sans-serif; }
    .title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .alert-box {
        background: #ff2d2d;
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    .safe-box {
        background: #1a1a1a;
        color: #00ff88;
        padding: 1rem 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1rem 0;
        border: 1px solid #00ff8844;
    }
</style>
""", unsafe_allow_html=True)

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class PhoneDetector(VideoProcessorBase):
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.phone_detected = False
        self.frame_count = 0
        self.lock = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % 3 == 0:
            results = self.model(img, verbose=False, conf=0.3, imgsz=640)
            detected = False

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    if class_name == 'cell phone':
                        detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(img, "PHONE DETECTED!", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            with self.lock:
                self.phone_detected = detected

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.markdown('<div class="title">🚫 No Phone Zone</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered phone detector — real time</div>', unsafe_allow_html=True)

status = st.empty()

ctx = webrtc_streamer(
    key="phone-detector",
    video_processor_factory=PhoneDetector,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.state.playing:
    while True:
        if ctx.video_processor:
            with ctx.video_processor.lock:
                detected = ctx.video_processor.phone_detected

            if detected:
                status.markdown('<div class="alert-box">📵 PHONE DETECTED — PUT IT DOWN!</div>', unsafe_allow_html=True)
            else:
                status.markdown('<div class="safe-box">✅ No phone detected — You\'re good!</div>', unsafe_allow_html=True)

        time.sleep(0.5)
