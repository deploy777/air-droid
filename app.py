import streamlit as st
import cv2
import av
import numpy as np
import time
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from utils import HandTracker, preprocess_gesture, draw_perfect_shape, heuristic_classify
from model import load_model, SHAPES

# ═══════════════════════════════════════════════════════════════
# Helper Functions for Improved Inference Pipeline
# (IDENTICAL to original app.py — no changes)
# ═══════════════════════════════════════════════════════════════

def smooth_points_ema(points, alpha=0.4):
    """Apply exponential moving average smoothing to reduce hand jitter."""
    if len(points) < 2:
        return points
    smoothed = [points[0]]
    for i in range(1, len(points)):
        sx = int(alpha * points[i][0] + (1 - alpha) * smoothed[-1][0])
        sy = int(alpha * points[i][1] + (1 - alpha) * smoothed[-1][1])
        smoothed.append((sx, sy))
    return smoothed


def remove_duplicate_points(points, min_dist=3):
    """Remove points that are within min_dist pixels of each other."""
    if len(points) < 2:
        return points
    filtered = [points[0]]
    for p in points[1:]:
        dx = p[0] - filtered[-1][0]
        dy = p[1] - filtered[-1][1]
        if (dx * dx + dy * dy) > min_dist * min_dist:
            filtered.append(p)
    return filtered


def interpolate_gaps(points, max_gap=20):
    """Interpolate between points that are far apart to fill gaps."""
    if len(points) < 2:
        return points
    interpolated = [points[0]]
    for i in range(1, len(points)):
        dx = points[i][0] - points[i-1][0]
        dy = points[i][1] - points[i-1][1]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > max_gap:
            num_interp = int(dist / max_gap)
            for j in range(1, num_interp + 1):
                t = j / (num_interp + 1)
                ix = int(points[i-1][0] + t * dx)
                iy = int(points[i-1][1] + t * dy)
                interpolated.append((ix, iy))
        interpolated.append(points[i])
    return interpolated


def test_time_augmentation(model, roi, shapes):
    """
    Run prediction on multiple augmented versions and average for robustness.
    Augmentations: rotations, slight scales, and gaussian noise.
    Horizontal flip is excluded — it corrupts asymmetric shapes.
    """
    h, w = roi.shape[1], roi.shape[2]
    center = (w // 2, h // 2)
    all_preds = []

    # Extract original image ONCE (avoid stale-variable bug)
    img = roi[0, :, :, 0]

    # 1) Rotations: -15, -10, -5, 0, +5, +10, +15
    for angle in [-15, -10, -5, 0, 5, 10, 15]:
        if angle == 0:
            aug_roi = roi
        else:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h))
            aug_roi = np.expand_dims(np.expand_dims(rotated, axis=-1), axis=0)
        preds = model.predict(aug_roi, verbose=0)
        all_preds.append(preds[0])

    # 2) Slight scale variations (0.9x and 1.1x)
    for scale in [0.9, 1.1]:
        M_scale = cv2.getRotationMatrix2D(center, 0, scale)
        scaled = cv2.warpAffine(img, M_scale, (w, h))
        aug_roi = np.expand_dims(np.expand_dims(scaled, axis=-1), axis=0)
        preds = model.predict(aug_roi, verbose=0)
        all_preds.append(preds[0])

    # 3) Gaussian noise variant
    noisy = np.clip(img + np.random.normal(0, 0.05, img.shape).astype(np.float32), 0, 1)
    aug_roi = np.expand_dims(np.expand_dims(noisy, axis=-1), axis=0)
    preds = model.predict(aug_roi, verbose=0)
    all_preds.append(preds[0])

    avg_preds = np.mean(all_preds, axis=0)
    class_idx = np.argmax(avg_preds)
    confidence = float(avg_preds[class_idx])
    shape_name = shapes[class_idx]

    # Get top-3 predictions
    top3_idx = np.argsort(avg_preds)[::-1][:3]
    top3 = [(shapes[i], float(avg_preds[i])) for i in top3_idx]

    return shape_name, confidence, top3


def geometric_disambiguate(points, cnn_shape, cnn_confidence, top3):
    """
    Use geometric features to disambiguate when CNN confidence is moderate.
    Resolves common confusions between visually similar shapes.
    """
    if cnn_confidence > 0.75 or len(points) < 15:
        return cnn_shape, cnn_confidence, top3

    pts = np.array(points, dtype=np.float32)
    # Compute geometric features
    x, y, bw, bh = cv2.boundingRect(pts.astype(np.int32))
    aspect = float(bw) / max(bh, 1)

    # Path properties
    start_end_dist = np.linalg.norm(pts[0] - pts[-1])
    max_extent = max(bw, bh)
    is_closed = start_end_dist < max_extent * 0.25

    # Check for self-crossings (characteristic of infinity, butterfly)
    crossings = 0
    segments = len(pts) - 1
    if segments > 10:
        step = max(1, segments // 30)
        for i in range(0, segments - 2, step):
            for j in range(i + 2, min(segments, i + 20), step):
                p1, p2 = pts[i], pts[i+1]
                p3, p4 = pts[j], pts[j+1]
                d1 = (p4[0]-p3[0])*(p1[1]-p3[1]) - (p4[1]-p3[1])*(p1[0]-p3[0])
                d2 = (p4[0]-p3[0])*(p2[1]-p3[1]) - (p4[1]-p3[1])*(p2[0]-p3[0])
                d3 = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                d4 = (p2[0]-p1[0])*(p4[1]-p1[1]) - (p2[1]-p1[1])*(p4[0]-p1[0])
                if d1 * d2 < 0 and d3 * d4 < 0:
                    crossings += 1

    # Radius progression (spiral check: monotonically increasing/decreasing)
    centroid = np.mean(pts, axis=0)
    radii = np.sqrt(np.sum((pts - centroid) ** 2, axis=1))
    radius_trend = np.corrcoef(np.arange(len(radii)), radii)[0, 1] if len(radii) > 2 else 0

    # Top-2 CNN candidates
    top2_names = [n for n, c in top3[:2]]

    # Disambiguation rules for common confusions
    # Spiral vs Infinity: spiral has monotonic radius, infinity crosses itself
    if set(top2_names) & {"Spiral", "Infinity"}:
        if crossings > 2 and abs(radius_trend) < 0.5:
            if cnn_shape != "Infinity":
                return "Infinity", max(cnn_confidence, 0.6), top3
        elif abs(radius_trend) > 0.6 and crossings <= 1:
            if cnn_shape != "Spiral":
                return "Spiral", max(cnn_confidence, 0.6), top3

    # Cloud vs Smiley: smiley is more circular, cloud is bumpy
    if set(top2_names) & {"Cloud", "Smiley face"}:
        if 0.8 < aspect < 1.25 and is_closed:
            # More likely smiley (round, closed)
            pass  # Trust CNN here

    # Lightning vs Crown: lightning is vertical, crown is horizontal
    if set(top2_names) & {"Lightning bolt", "Crown"}:
        if aspect < 0.6:  # Tall and narrow
            if cnn_shape != "Lightning bolt":
                return "Lightning bolt", max(cnn_confidence, 0.6), top3
        elif aspect > 1.5:  # Wide
            if cnn_shape != "Crown":
                return "Crown", max(cnn_confidence, 0.6), top3

    return cnn_shape, cnn_confidence, top3


def stabilize_drawing_point(points, window=3):
    """Average the last `window` points for smoother on-screen drawing."""
    if len(points) < window:
        window = len(points)
    recent = points[-window:]
    sx = int(np.mean([p[0] for p in recent]))
    sy = int(np.mean([p[1] for p in recent]))
    return (sx, sy)


# ═══════════════════════════════════════════════════════════════
# Page Config & Styling
# ═══════════════════════════════════════════════════════════════

st.set_page_config(page_title="AI Air Drawing Shape Detector", layout="wide")

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        color: #ffffff;
    }
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
    }
    .stSidebar {
        background-color: rgba(45, 45, 68, 0.8);
        border-right: 1px solid #444;
    }
    h1 {
        color: #00f2fe;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(45deg, #00dbde 0%, #fc00ff 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(252, 0, 255, 0.4);
    }
    .shape-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
        border-left: 5px solid #00f2fe;
    }
    .prediction-card {
        background: rgba(0, 242, 254, 0.1);
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 4px;
        border-left: 4px solid #fc00ff;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎨 AI Air Canvas — Creative Shape Detector")
st.subheader("Draw & detect 12 creative shapes in real-time using CNN + Transformer AI")

# Show supported shapes
shape_emojis = {
    "Spiral": "🌀", "Infinity": "♾️", "Cloud": "☁️", "Lightning bolt": "⚡",
    "Flower": "🌸", "Butterfly": "🦋", "Crown": "👑", "Flame": "🔥",
    "Fish": "🐟", "Leaf": "🍃", "Music note": "🎵", "Smiley face": "😊"
}

# Initialize Session State
if 'detected_shapes' not in st.session_state:
    st.session_state.detected_shapes = []
if 'current_color' not in st.session_state:
    st.session_state.current_color = (0, 255, 0)
if 'top_predictions' not in st.session_state:
    st.session_state.top_predictions = []
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# Sidebar
st.sidebar.header("🎮 Controls")
color_picker = st.sidebar.color_picker("Pick a Shape Fill Color", "#00FF00")
hex_color = color_picker.lstrip('#')
st.session_state.current_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

st.sidebar.markdown("---")
demo_mode = st.sidebar.checkbox("Use Heuristic Fallback", value=False,
    help="Uses geometric heuristics instead of the AI model. AI model is more advanced (CNN+Transformer).")
use_tta = st.sidebar.checkbox("Use Test-Time Augmentation", value=True,
    help="Averages multiple predictions for more robust classification. Slightly slower but more accurate.")

if st.sidebar.button("🧹 Clear Canvas"):
    st.session_state.detected_shapes = []
    st.session_state.top_predictions = []
    st.session_state.last_result = None

if st.sidebar.button("↩️ Undo Last Shape"):
    if st.session_state.detected_shapes:
        st.session_state.detected_shapes.pop()
        st.session_state.top_predictions = []

st.sidebar.markdown("---")
st.sidebar.header("🎯 Supported Shapes")
shape_cols = st.sidebar.columns(2)
for i, shape in enumerate(SHAPES):
    emoji = shape_emojis.get(shape, "🔹")
    shape_cols[i % 2].markdown(f"{emoji} {shape}")

st.sidebar.markdown("---")
st.sidebar.header("📜 Detected Shapes")
for i, shape in enumerate(st.session_state.detected_shapes):
    emoji = shape_emojis.get(shape['name'], "🔹")
    conf_str = f" ({shape.get('confidence', 0):.0%})" if 'confidence' in shape else ""
    st.sidebar.markdown(f"""
        <div class="shape-card">
            <strong>{emoji} {i+1}. {shape['name']}{conf_str}</strong>
        </div>
    """, unsafe_allow_html=True)

# Show top-3 predictions
if st.session_state.top_predictions:
    st.sidebar.markdown("---")
    st.sidebar.header("🔮 Last Prediction (Top 3)")
    for rank, (name, conf) in enumerate(st.session_state.top_predictions):
        emoji = shape_emojis.get(name, "🔹")
        bar = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
        st.sidebar.markdown(f"""
            <div class="prediction-card">
                <strong>#{rank+1} {emoji} {name}</strong><br>
                <code>{bar}</code> {conf:.1%}
            </div>
        """, unsafe_allow_html=True)

# AI Model
@st.cache_resource
def get_ai_model():
    return load_model()

model = get_ai_model()

# ═══════════════════════════════════════════════════════════════
# Shared State for WebRTC callback thread
# ═══════════════════════════════════════════════════════════════

CLASSIFY_DELAY = 0.5
MIN_POINTS = 20
CONFIDENCE_ACCEPT = 0.55
CONFIDENCE_UNCERTAIN = 0.35

# Thread-safe shared state
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.points = []
        self.last_point = None
        self.hand_disappeared_time = None
        self.detected_shapes = []
        self.result = None
        self.top_predictions = []
        self.drawing_color = (0, 255, 0)

shared = SharedState()

# ═══════════════════════════════════════════════════════════════
# WebRTC Video Frame Callback (runs in separate thread)
# ═══════════════════════════════════════════════════════════════

# Lazy-init tracker (only created once, not on every Streamlit rerun)
_tracker_lock = threading.Lock()
_tracker_instance = None

def _get_tracker():
    global _tracker_instance
    if _tracker_instance is None:
        with _tracker_lock:
            if _tracker_instance is None:
                _tracker_instance = HandTracker()
    return _tracker_instance

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w, c = img.shape

    with shared.lock:
        current_color = shared.drawing_color

    # Hand Detection (lazy init)
    tracker = _get_tracker()
    results = tracker.find_hand_landmarks(img)
    cx, cy, landmarks = tracker.get_finger_tip(results, w, h)

    if landmarks:
        with shared.lock:
            shared.hand_disappeared_time = None

            # Point deduplication
            should_add = True
            if shared.last_point is not None:
                dx = cx - shared.last_point[0]
                dy = cy - shared.last_point[1]
                if (dx * dx + dy * dy) < 4:
                    should_add = False

            if should_add:
                shared.points.append((cx, cy))
                shared.last_point = (cx, cy)

            display_pt = stabilize_drawing_point(shared.points)

        cv2.circle(img, display_pt, 15, current_color, -1)
        cv2.putText(img, "Drawing...", (display_pt[0] + 20, display_pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tracker.mp_draw.draw_landmarks(img, landmarks, tracker.mp_hands.HAND_CONNECTIONS)

    else:
        with shared.lock:
            num_points = len(shared.points)
            hand_time = shared.hand_disappeared_time

        if num_points > MIN_POINTS:
            if hand_time is None:
                with shared.lock:
                    shared.hand_disappeared_time = time.time()
                hand_time = time.time()

            elapsed = time.time() - hand_time

            if elapsed >= CLASSIFY_DELAY:
                with shared.lock:
                    pts_copy = shared.points.copy()
                    shared.points = []
                    shared.last_point = None
                    shared.hand_disappeared_time = None

                # Preprocess points
                processed_pts = smooth_points_ema(pts_copy, alpha=0.5)
                processed_pts = remove_duplicate_points(processed_pts)
                processed_pts = interpolate_gaps(processed_pts)

                roi = preprocess_gesture(processed_pts, frame_hw=(h, w))
                if roi is not None:
                    if demo_mode:
                        shape_name, confidence = heuristic_classify(processed_pts, frame_hw=(h, w))
                        top3 = [(shape_name, confidence)]
                    else:
                        if use_tta:
                            shape_name, confidence, top3 = test_time_augmentation(
                                model, roi, SHAPES
                            )
                        else:
                            preds = model.predict(roi, verbose=0)
                            class_idx = np.argmax(preds[0])
                            confidence = float(np.max(preds[0]))
                            shape_name = SHAPES[class_idx]
                            top3_idx = np.argsort(preds[0])[::-1][:3]
                            top3 = [(SHAPES[i], float(preds[0][i])) for i in top3_idx]

                        shape_name, confidence, top3 = geometric_disambiguate(
                            processed_pts, shape_name, confidence, top3
                        )

                    with shared.lock:
                        shared.top_predictions = top3

                        emoji = shape_emojis.get(shape_name, "🔹")
                        if confidence >= CONFIDENCE_ACCEPT:
                            shared.detected_shapes.append({
                                "name": shape_name,
                                "points": pts_copy,
                                "color": current_color,
                                "confidence": confidence
                            })
                            shared.result = ("success", f"{emoji} Detected: **{shape_name}** ({confidence:.1%})", confidence)
                        elif confidence >= CONFIDENCE_UNCERTAIN:
                            top2_str = " / ".join(
                                [f"{shape_emojis.get(n, '')} {n} ({c:.0%})" for n, c in top3[:2]]
                            )
                            shared.result = ("warning", f"🤔 Uncertain: {top2_str}", confidence)
                        else:
                            shared.result = ("error", "❓ Unknown shape — try drawing more clearly", confidence)
                else:
                    with shared.lock:
                        shared.result = ("error", "❓ Drawing too small — try drawing larger", 0)
        else:
            with shared.lock:
                if shared.hand_disappeared_time is not None:
                    elapsed = time.time() - shared.hand_disappeared_time
                    if elapsed >= CLASSIFY_DELAY + 0.5:
                        shared.points = []
                        shared.last_point = None
                        shared.hand_disappeared_time = None

    # Draw current points
    with shared.lock:
        pts_draw = shared.points.copy()
        shapes_draw = shared.detected_shapes.copy()

    for i in range(1, len(pts_draw)):
        cv2.line(img, pts_draw[i-1], pts_draw[i], current_color, 4)

    # Draw all previously detected shapes
    for shape in shapes_draw:
        img = draw_perfect_shape(img, shape['name'], shape['color'], shape['points'])

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# ═══════════════════════════════════════════════════════════════
# Main Layout
# ═══════════════════════════════════════════════════════════════

col1, col2 = st.columns([3, 1])

with col1:
    ctx = webrtc_streamer(
        key="air-canvas",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        async_processing=True,
    )

with col2:
    st.info("**How to use:**\n1. Click START to enable camera\n2. Use your **index finger** to draw\n3. Keep other fingers closed\n4. When you stop moving or move away, AI classifies the shape\n\n**Supported shapes:**\n" + ", ".join([f"{shape_emojis.get(s, '')} {s}" for s in SHAPES]))
    prediction_label = st.empty()
    confidence_bar = st.empty()

# Poll shared state for results (no st.rerun to avoid overload)
if ctx.state.playing:
    with shared.lock:
        result = shared.result
        top_preds = shared.top_predictions.copy()
        det_shapes = shared.detected_shapes.copy()
        shared.drawing_color = st.session_state.current_color

    if result:
        level, msg, conf = result
        if level == "success":
            prediction_label.success(msg)
        elif level == "warning":
            prediction_label.warning(msg)
        else:
            prediction_label.error(msg)
        confidence_bar.progress(min(conf, 1.0))

    st.session_state.top_predictions = top_preds
    st.session_state.detected_shapes = det_shapes
