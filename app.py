"""
AI Air Drawing — Flask version
Same model, same inference pipeline as app.py — just HTML instead of Streamlit.
Uses server-side OpenCV + MediaPipe (identical to original app.py).
Video is streamed to browser via MJPEG.
"""

import cv2
import numpy as np
import time
import json
import threading
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
from utils import HandTracker, preprocess_gesture, draw_perfect_shape, heuristic_classify
from model import load_model, SHAPES

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# ── Load AI model (same as app.py) ──
print("Loading AI model...")
model = load_model()
print(f"Model loaded! Input: {model.input_shape} -> Output: {model.output_shape}")

tracker = HandTracker()

# ── Shared state (thread-safe) ──
state_lock = threading.Lock()
state = {
    "run_camera": False,
    "points": [],
    "detected_shapes": [],
    "current_color": (0, 255, 136),
    "hand_disappeared_time": None,
    "top_predictions": [],
    "last_point": None,
    "last_result": None,
}

SHAPE_EMOJIS = {
    "Spiral": "🌀", "Infinity": "♾️", "Cloud": "☁️", "Lightning bolt": "⚡",
    "Flower": "🌸", "Butterfly": "🦋", "Crown": "👑", "Flame": "🔥",
    "Fish": "🐟", "Leaf": "🍃", "Music note": "🎵", "Smiley face": "😊"
}

# ═══════════════════════════════════════════════════════════════
# Helper Functions — IDENTICAL to app.py
# ═══════════════════════════════════════════════════════════════

def smooth_points_ema(points, alpha=0.4):
    if len(points) < 2:
        return points
    smoothed = [points[0]]
    for i in range(1, len(points)):
        sx = int(alpha * points[i][0] + (1 - alpha) * smoothed[-1][0])
        sy = int(alpha * points[i][1] + (1 - alpha) * smoothed[-1][1])
        smoothed.append((sx, sy))
    return smoothed


def remove_duplicate_points(points, min_dist=3):
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
    h, w = roi.shape[1], roi.shape[2]
    center = (w // 2, h // 2)
    all_preds = []
    img = roi[0, :, :, 0]

    for angle in [-15, -10, -5, 0, 5, 10, 15]:
        if angle == 0:
            aug_roi = roi
        else:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h))
            aug_roi = np.expand_dims(np.expand_dims(rotated, axis=-1), axis=0)
        preds = model.predict(aug_roi, verbose=0)
        all_preds.append(preds[0])

    for scale in [0.9, 1.1]:
        M_scale = cv2.getRotationMatrix2D(center, 0, scale)
        scaled = cv2.warpAffine(img, M_scale, (w, h))
        aug_roi = np.expand_dims(np.expand_dims(scaled, axis=-1), axis=0)
        preds = model.predict(aug_roi, verbose=0)
        all_preds.append(preds[0])

    noisy = np.clip(img + np.random.normal(0, 0.05, img.shape).astype(np.float32), 0, 1)
    aug_roi = np.expand_dims(np.expand_dims(noisy, axis=-1), axis=0)
    preds = model.predict(aug_roi, verbose=0)
    all_preds.append(preds[0])

    avg_preds = np.mean(all_preds, axis=0)
    class_idx = np.argmax(avg_preds)
    confidence = float(avg_preds[class_idx])
    shape_name = shapes[class_idx]

    top3_idx = np.argsort(avg_preds)[::-1][:3]
    top3 = [(shapes[i], float(avg_preds[i])) for i in top3_idx]

    return shape_name, confidence, top3


def geometric_disambiguate(points, cnn_shape, cnn_confidence, top3):
    if cnn_confidence > 0.75 or len(points) < 15:
        return cnn_shape, cnn_confidence, top3

    pts = np.array(points, dtype=np.float32)
    x, y, bw, bh = cv2.boundingRect(pts.astype(np.int32))
    aspect = float(bw) / max(bh, 1)

    start_end_dist = np.linalg.norm(pts[0] - pts[-1])
    max_extent = max(bw, bh)
    is_closed = start_end_dist < max_extent * 0.25

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

    centroid = np.mean(pts, axis=0)
    radii = np.sqrt(np.sum((pts - centroid) ** 2, axis=1))
    radius_trend = np.corrcoef(np.arange(len(radii)), radii)[0, 1] if len(radii) > 2 else 0

    top2_names = [n for n, c in top3[:2]]

    if set(top2_names) & {"Spiral", "Infinity"}:
        if crossings > 2 and abs(radius_trend) < 0.5:
            if cnn_shape != "Infinity":
                return "Infinity", max(cnn_confidence, 0.6), top3
        elif abs(radius_trend) > 0.6 and crossings <= 1:
            if cnn_shape != "Spiral":
                return "Spiral", max(cnn_confidence, 0.6), top3

    if set(top2_names) & {"Lightning bolt", "Crown"}:
        if aspect < 0.6:
            if cnn_shape != "Lightning bolt":
                return "Lightning bolt", max(cnn_confidence, 0.6), top3
        elif aspect > 1.5:
            if cnn_shape != "Crown":
                return "Crown", max(cnn_confidence, 0.6), top3

    return cnn_shape, cnn_confidence, top3


def stabilize_drawing_point(points, window=3):
    if len(points) < window:
        window = len(points)
    recent = points[-window:]
    sx = int(np.mean([p[0] for p in recent]))
    sy = int(np.mean([p[1] for p in recent]))
    return (sx, sy)


# ═══════════════════════════════════════════════════════════════
# Camera Loop — same logic as app.py, runs in background thread
# ═══════════════════════════════════════════════════════════════

CLASSIFY_DELAY = 0.5
MIN_POINTS = 20
CONFIDENCE_ACCEPT = 0.55
CONFIDENCE_UNCERTAIN = 0.35

output_frame = None
frame_lock = threading.Lock()


def camera_loop():
    global output_frame
    cap = cv2.VideoCapture(0)

    while True:
        with state_lock:
            if not state["run_camera"]:
                time.sleep(0.1)
                continue

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        results = tracker.find_hand_landmarks(frame)
        cx, cy, landmarks = tracker.get_finger_tip(results, w, h)

        with state_lock:
            if landmarks:
                state["hand_disappeared_time"] = None

                should_add = True
                if state["last_point"] is not None:
                    dx = cx - state["last_point"][0]
                    dy = cy - state["last_point"][1]
                    if (dx * dx + dy * dy) < 4:
                        should_add = False

                if should_add:
                    state["points"].append((cx, cy))
                    state["last_point"] = (cx, cy)

                display_pt = stabilize_drawing_point(state["points"])
                cv2.circle(frame, display_pt, 15, state["current_color"], -1)
                cv2.putText(frame, "Drawing...", (display_pt[0] + 20, display_pt[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                tracker.mp_draw.draw_landmarks(frame, landmarks, tracker.mp_hands.HAND_CONNECTIONS)
            else:
                if len(state["points"]) > MIN_POINTS:
                    if state["hand_disappeared_time"] is None:
                        state["hand_disappeared_time"] = time.time()

                    elapsed = time.time() - state["hand_disappeared_time"]

                    if elapsed >= CLASSIFY_DELAY:
                        processed_pts = smooth_points_ema(state["points"], alpha=0.5)
                        processed_pts = remove_duplicate_points(processed_pts)
                        processed_pts = interpolate_gaps(processed_pts)

                        roi = preprocess_gesture(processed_pts, frame_hw=(h, w))
                        if roi is not None:
                            shape_name, confidence, top3 = test_time_augmentation(model, roi, SHAPES)
                            shape_name, confidence, top3 = geometric_disambiguate(
                                processed_pts, shape_name, confidence, top3
                            )

                            state["top_predictions"] = top3
                            emoji = SHAPE_EMOJIS.get(shape_name, "")

                            if confidence >= CONFIDENCE_ACCEPT:
                                state["detected_shapes"].append({
                                    "name": shape_name,
                                    "points": state["points"].copy(),
                                    "color": state["current_color"],
                                    "confidence": confidence
                                })
                                state["last_result"] = {
                                    "shape": shape_name,
                                    "emoji": emoji,
                                    "confidence": round(confidence * 100, 1),
                                    "top3": [
                                        {"name": n, "emoji": SHAPE_EMOJIS.get(n, ""), "confidence": round(c * 100, 1)}
                                        for n, c in top3
                                    ],
                                    "accepted": True,
                                    "status": "accepted"
                                }
                            elif confidence >= CONFIDENCE_UNCERTAIN:
                                state["last_result"] = {
                                    "shape": shape_name,
                                    "emoji": emoji,
                                    "confidence": round(confidence * 100, 1),
                                    "top3": [
                                        {"name": n, "emoji": SHAPE_EMOJIS.get(n, ""), "confidence": round(c * 100, 1)}
                                        for n, c in top3
                                    ],
                                    "accepted": False,
                                    "status": "uncertain"
                                }
                            else:
                                state["last_result"] = {
                                    "shape": "Unknown",
                                    "emoji": "?",
                                    "confidence": round(confidence * 100, 1),
                                    "top3": [],
                                    "accepted": False,
                                    "status": "unknown"
                                }

                        state["points"] = []
                        state["last_point"] = None
                        state["hand_disappeared_time"] = None
                else:
                    if state["hand_disappeared_time"] is not None:
                        elapsed = time.time() - state["hand_disappeared_time"]
                        if elapsed >= CLASSIFY_DELAY + 0.5:
                            state["points"] = []
                            state["last_point"] = None
                            state["hand_disappeared_time"] = None

            # Draw current points
            for i in range(1, len(state["points"])):
                cv2.line(frame, state["points"][i-1], state["points"][i],
                         state["current_color"], 4)

            # Draw detected shapes
            for shape in state["detected_shapes"]:
                frame = draw_perfect_shape(frame, shape['name'], shape['color'], shape['points'])

        # Encode frame as JPEG for streaming
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            output_frame = buffer.tobytes()

        time.sleep(0.01)

    cap.release()


# ═══════════════════════════════════════════════════════════════
# Flask Routes
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


def generate_frames():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                time.sleep(0.05)
                continue
            frame = output_frame

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/api/start", methods=["POST"])
def start_camera():
    with state_lock:
        state["run_camera"] = True
    return jsonify({"status": "started"})


@app.route("/api/stop", methods=["POST"])
def stop_camera():
    with state_lock:
        state["run_camera"] = False
    return jsonify({"status": "stopped"})


@app.route("/api/clear", methods=["POST"])
def clear_canvas():
    with state_lock:
        state["points"] = []
        state["detected_shapes"] = []
        state["top_predictions"] = []
        state["last_point"] = None
        state["last_result"] = None
    return jsonify({"status": "cleared"})


@app.route("/api/undo", methods=["POST"])
def undo_shape():
    with state_lock:
        if state["detected_shapes"]:
            state["detected_shapes"].pop()
    return jsonify({"status": "undone", "remaining": len(state["detected_shapes"])})


@app.route("/api/color", methods=["POST"])
def set_color():
    data = request.get_json()
    hex_color = data.get("color", "#00FF88").lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    with state_lock:
        state["current_color"] = (r, g, b)
    return jsonify({"status": "ok"})


@app.route("/api/state")
def get_state():
    with state_lock:
        return jsonify({
            "running": state["run_camera"],
            "num_points": len(state["points"]),
            "num_shapes": len(state["detected_shapes"]),
            "shapes": [
                {
                    "name": s["name"],
                    "confidence": round(s.get("confidence", 0) * 100, 1),
                    "emoji": SHAPE_EMOJIS.get(s["name"], "")
                }
                for s in state["detected_shapes"]
            ],
            "last_result": state["last_result"],
            "top_predictions": [
                {"name": n, "emoji": SHAPE_EMOJIS.get(n, ""), "confidence": round(c * 100, 1)}
                for n, c in state.get("top_predictions", [])
            ]
        })


if __name__ == "__main__":
    # Start camera thread
    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
