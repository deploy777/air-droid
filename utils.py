import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hand_landmarks(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        return results

    def get_finger_tip(self, results, width, height):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get landmarks
                lm = hand_landmarks.landmark
                
                # Finger states (Up or Down)
                # Tips: 8 (Index), 12 (Middle), 16 (Ring), 20 (Pinky)
                # Lower joints: 6 (Index), 10 (Middle), 14 (Ring), 18 (Pinky)
                
                index_up = lm[8].y < lm[6].y
                middle_up = lm[12].y < lm[10].y
                ring_up = lm[16].y < lm[14].y
                pinky_up = lm[20].y < lm[18].y
                
                # Drawing gesture: Index up, others down
                if index_up and not middle_up and not ring_up and not pinky_up:
                    cx, cy = int(lm[8].x * width), int(lm[8].y * height)
                    return cx, cy, hand_landmarks
        return None, None, None

def preprocess_canvas(mask, output_size=128):
    """
    Shared preprocessing: crop drawing to content, pad to square, resize.
    Used by BOTH training data generation AND inference to guarantee alignment.

    Args:
        mask: Binary image (uint8, white-on-black) of any size.
        output_size: Target square dimension.
    Returns:
        Processed uint8 image (output_size x output_size), or None if empty/too small.
    """
    nonzero = cv2.findNonZero(mask)
    if nonzero is None:
        return None

    x, y, w, h = cv2.boundingRect(nonzero)

    if w < 10 or h < 10:
        return None

    # Proportional padding — track actual shift to avoid asymmetric overpad at edges
    pad = max(15, max(w, h) // 6)
    orig_x, orig_y = x, y
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(mask.shape[1] - x, w + (orig_x - x) + pad)
    h = min(mask.shape[0] - y, h + (orig_y - y) + pad)

    roi = mask[y:y+h, x:x+w]

    # Pad to square preserving aspect ratio
    max_dim = max(roi.shape[0], roi.shape[1])
    square = np.zeros((max_dim, max_dim), dtype=np.uint8)
    y_off = (max_dim - roi.shape[0]) // 2
    x_off = (max_dim - roi.shape[1]) // 2
    square[y_off:y_off+roi.shape[0], x_off:x_off+roi.shape[1]] = roi

    result = cv2.resize(square, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return result


def preprocess_gesture(points, frame_hw=(480, 640), canvas_size=(128, 128)):
    """
    Preprocesses drawn points into model input using the shared preprocess_canvas.
    Accepts actual frame dimensions to handle any webcam resolution.
    """
    if not points or len(points) < 2:
        return None

    mask = np.zeros(frame_hw, dtype=np.uint8)

    thickness = 4
    for i in range(1, len(points)):
        if points[i-1] is not None and points[i] is not None:
            cv2.line(mask, points[i-1], points[i], 255, thickness)

    result = preprocess_canvas(mask, output_size=canvas_size[0])
    if result is None:
        return None

    result = result.astype('float32') / 255.0
    result = np.expand_dims(result, axis=-1)
    result = np.expand_dims(result, axis=0)
    return result

def heuristic_classify(points, frame_hw=(480, 640)):
    """
    Heuristic classification fallback for the 12 creative shapes.
    Uses geometric properties, path analysis, and contour features.
    """
    if not points or len(points) < 10:
        return "Spiral", 0.4

    mask = np.zeros(frame_hw, dtype=np.uint8)
    for i in range(1, len(points)):
        if points[i-1] is not None and points[i] is not None:
            cv2.line(mask, points[i-1], points[i], 255, 4)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Spiral", 0.4

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)

    if area < 100:
        return "Spiral", 0.4

    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    num_vertices = len(approx)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 1

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 1
    circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0

    num_contours = len(contours)

    pts_array = np.array(points, dtype=np.float32)
    start_end_dist = np.linalg.norm(pts_array[0] - pts_array[-1])
    max_extent = max(w, h)
    is_open = start_end_dist > max_extent * 0.25
    is_closed = not is_open

    # Path-based features
    # Radius progression (spiral detection: monotonic radius change)
    centroid = np.mean(pts_array, axis=0)
    radii = np.sqrt(np.sum((pts_array - centroid) ** 2, axis=1))
    radius_trend = 0.0
    if len(radii) > 5:
        radius_trend = float(np.corrcoef(np.arange(len(radii)), radii)[0, 1])

    # Self-crossing count (infinity, butterfly detection)
    crossings = 0
    n_pts = len(pts_array)
    if n_pts > 10:
        step = max(1, n_pts // 40)
        for i in range(0, n_pts - 3, step):
            for j in range(i + 3, min(n_pts - 1, i + 25), step):
                p1, p2 = pts_array[i], pts_array[i+1]
                p3, p4 = pts_array[j], pts_array[j+1]
                d1 = (p4[0]-p3[0])*(p1[1]-p3[1]) - (p4[1]-p3[1])*(p1[0]-p3[0])
                d2 = (p4[0]-p3[0])*(p2[1]-p3[1]) - (p4[1]-p3[1])*(p2[0]-p3[0])
                d3 = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                d4 = (p2[0]-p1[0])*(p4[1]-p1[1]) - (p2[1]-p1[1])*(p4[0]-p1[0])
                if d1 * d2 < 0 and d3 * d4 < 0:
                    crossings += 1

    # Angular change analysis (zigzag detection for lightning, crown)
    angles = []
    step_a = max(1, n_pts // 30)
    for i in range(step_a, n_pts - step_a, step_a):
        v1 = pts_array[i] - pts_array[i - step_a]
        v2 = pts_array[i + step_a] - pts_array[i]
        dot = np.dot(v1, v2)
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        angle = abs(np.arctan2(cross, dot))
        angles.append(angle)
    sharp_turns = sum(1 for a in angles if a > np.pi * 0.4)

    # Decision tree with discriminative features

    # Spiral: open path with monotonically changing radius
    if is_open and abs(radius_trend) > 0.5 and crossings < 3:
        return "Spiral", 0.75

    # Infinity: closed-ish, has self-crossings, wider than tall
    if crossings > 3 and aspect_ratio > 1.2 and abs(radius_trend) < 0.4:
        return "Infinity", 0.7

    # Lightning bolt: open, vertical, sharp zigzag turns
    if is_open and sharp_turns >= 2 and (aspect_ratio < 0.6 or aspect_ratio > 1.8):
        return "Lightning bolt", 0.7

    # Smiley face: round, closed, high circularity
    if circularity > 0.5 and is_closed and 0.7 < aspect_ratio < 1.4:
        return "Smiley face", 0.65

    # Crown: wide, has sharp peaks on top, closed
    if aspect_ratio > 1.2 and sharp_turns >= 3 and is_closed:
        return "Crown", 0.6

    # Music note: open, has a round part and a line part
    if is_open and aspect_ratio < 1.0 and n_pts > 30:
        return "Music note", 0.55

    # Butterfly: has 4-way symmetry with crossings
    if crossings > 1 and 0.7 < aspect_ratio < 1.5 and solidity < 0.6:
        return "Butterfly", 0.6

    # Flower: lobed/bumpy closed curve
    if is_closed and num_vertices > 8 and solidity < 0.7:
        return "Flower", 0.6

    # Fish: elongated, closed
    if aspect_ratio > 1.5 and is_closed and solidity > 0.5:
        return "Fish", 0.55

    # Leaf: elongated, closed, high solidity
    if (aspect_ratio > 1.3 or aspect_ratio < 0.7) and solidity > 0.65:
        return "Leaf", 0.55

    # Flame: taller than wide, tapered
    if aspect_ratio < 0.8 and solidity > 0.5:
        return "Flame", 0.55

    # Cloud: bumpy closed curve, moderate circularity
    if is_closed and circularity > 0.25 and solidity > 0.55:
        return "Cloud", 0.5

    return "Spiral", 0.4


def draw_perfect_shape(image, shape_name, color, points):
    """
    Renders a clean version of the detected shape overlay.
    Only handles the 12 creative shapes.
    """
    if not points:
        return image
    
    # Calculate bounding box of original points to place the perfect shape
    pts = np.array([p for p in points if p is not None])
    if len(pts) == 0: return image
    
    x, y, w, h = cv2.boundingRect(pts)
    center = (x + w // 2, y + h // 2)
    
    overlay = image.copy()
    color_bgr = (color[2], color[1], color[0]) # RGB to BGR
    
    if shape_name == "Spiral":
        pts_draw = []
        for theta in np.arange(0, 4 * np.pi, 0.1):
            r = (theta / (4 * np.pi)) * (min(w, h)/2)
            px = int(center[0] + r * np.cos(theta))
            py = int(center[1] + r * np.sin(theta))
            pts_draw.append([px, py])
        cv2.polylines(overlay, [np.array(pts_draw)], False, color_bgr, 5)
    
    elif shape_name == "Infinity":
        pts_draw = []
        for t in np.arange(0, 2 * np.pi, 0.1):
            den = 1 + np.sin(t)**2
            scale = min(w, h) / 2
            px = int(center[0] + scale * np.cos(t) / den * 1.5)
            py = int(center[1] + scale * np.sin(t) * np.cos(t) / den * 1.5)
            pts_draw.append([px, py])
        cv2.polylines(overlay, [np.array(pts_draw)], True, color_bgr, 5)
    
    elif shape_name == "Cloud":
        r = int(min(w, h) * 0.4)
        circles = [
            (center[0], center[1], r),
            (center[0]-int(r*0.8), center[1]+int(r*0.3), int(r*0.7)),
            (center[0]+int(r*0.8), center[1]+int(r*0.3), int(r*0.7)),
            (center[0], center[1]-int(r*0.5), int(r*0.6))
        ]
        for (cx, cy, cr) in circles:
            cv2.circle(overlay, (cx, cy), cr, color_bgr, -1)
    
    elif shape_name == "Lightning bolt":
        pts_draw = np.array([
            [center[0]-w//4, center[1]-h//2],
            [center[0]+w//4, center[1]-h//6],
            [center[0], center[1]-h//6],
            [center[0]+w//4, center[1]+h//2],
            [center[0]-w//4, center[1]+h//6],
            [center[0], center[1]+h//6]
        ])
        cv2.drawContours(overlay, [pts_draw], 0, color_bgr, -1)
    
    elif shape_name == "Flower":
        cv2.circle(overlay, center, min(w,h)//4, color_bgr, -1)
        for i in range(5):
            angle = i * 2 * np.pi / 5
            px = int(center[0] + (min(w,h)//2) * np.cos(angle))
            py = int(center[1] + (min(w,h)//2) * np.sin(angle))
            cv2.circle(overlay, (px, py), min(w,h)//4, color_bgr, -1)
    
    elif shape_name == "Butterfly":
        cv2.ellipse(overlay, center, (w//8, h//2), 0, 0, 360, color_bgr, -1)
        cv2.ellipse(overlay, (center[0]-w//4, center[1]-h//4), (w//3, h//3), 30, 0, 360, color_bgr, -1)
        cv2.ellipse(overlay, (center[0]+w//4, center[1]-h//4), (w//3, h//3), -30, 0, 360, color_bgr, -1)
        cv2.ellipse(overlay, (center[0]-w//4, center[1]+h//4), (w//4, h//4), -30, 0, 360, color_bgr, -1)
        cv2.ellipse(overlay, (center[0]+w//4, center[1]+h//4), (w//4, h//4), 30, 0, 360, color_bgr, -1)
    
    elif shape_name == "Crown":
        pts_draw = np.array([
            [center[0]-w//2, center[1]+h//2],
            [center[0]-w//2, center[1]-h//4],
            [center[0]-w//4, center[1]],
            [center[0], center[1]-h//2],
            [center[0]+w//4, center[1]],
            [center[0]+w//2, center[1]-h//4],
            [center[0]+w//2, center[1]+h//2]
        ])
        cv2.drawContours(overlay, [pts_draw], 0, color_bgr, -1)
    
    elif shape_name == "Flame":
        pts_draw = []
        for t in np.arange(0, 2*np.pi, 0.1):
            fx = int(center[0] + (w/2) * np.cos(t))
            fy = int(center[1] + (h/2) * np.sin(t) * (np.sin(t/2)**0.5 if t < np.pi else 1))
            pts_draw.append([fx, fy])
        cv2.drawContours(overlay, [np.array(pts_draw)], 0, color_bgr, -1)
    
    elif shape_name == "Fish":
        cv2.ellipse(overlay, center, (w//2, h//3), 0, 0, 360, color_bgr, -1)
        tail_pts = np.array([
            [center[0]-w//2, center[1]],
            [center[0]-w, center[1]-h//4],
            [center[0]-w, center[1]+h//4]
        ])
        cv2.drawContours(overlay, [tail_pts], 0, color_bgr, -1)
    
    elif shape_name == "Leaf":
        pts_draw = []
        for t in np.linspace(0, np.pi, 20):
            pts_draw.append([int(center[0]-w//2 + w*t/np.pi), int(center[1] - (h//2)*np.sin(t))])
        for t in np.linspace(0, np.pi, 20):
            pts_draw.append([int(center[0]+w//2 - w*t/np.pi), int(center[1] + (h//2)*np.sin(t))])
        cv2.drawContours(overlay, [np.array(pts_draw)], 0, color_bgr, -1)
    
    elif shape_name == "Music note":
        cv2.circle(overlay, (center[0]-w//4, center[1]+h//4), w//6, color_bgr, -1)
        cv2.line(overlay, (center[0]-w//4 + w//6, center[1]+h//4), (center[0]-w//4 + w//6, center[1]-h//2), color_bgr, 5)
        cv2.line(overlay, (center[0]-w//4 + w//6, center[1]-h//2), (center[0]+w//4, center[1]-h//4), color_bgr, 5)
    
    elif shape_name == "Smiley face":
        cv2.circle(overlay, center, min(w,h)//2, color_bgr, 5)
        cv2.circle(overlay, (center[0]-w//6, center[1]-w//6), w//20, color_bgr, -1)
        cv2.circle(overlay, (center[0]+w//6, center[1]-w//6), w//20, color_bgr, -1)
        cv2.ellipse(overlay, (center[0], center[1]+w//10), (w//4, w//6), 0, 0, 180, color_bgr, 5)
    
    else:
        # Fallback: just draw a highlight around the drawn points
        cv2.polylines(overlay, [pts], False, color_bgr, 4)

    # Transparency
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Add text
    cv2.putText(image, shape_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return image
