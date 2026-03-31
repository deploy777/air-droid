import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
import math
import random
from collections import Counter
from utils import preprocess_canvas


# ═══════════════════════════════════════════════════════════════
# Custom Layers for Enhanced CNN + Vision Transformer
# ═══════════════════════════════════════════════════════════════

class SqueezeExcitation(layers.Layer):
    """Channel attention via Squeeze-and-Excitation."""
    def __init__(self, filters, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio

    def build(self, input_shape):
        self.gap = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(self.filters // self.ratio, activation='relu')
        self.dense2 = layers.Dense(self.filters, activation='sigmoid')
        self.reshape = layers.Reshape((1, 1, self.filters))
        super().build(input_shape)

    def call(self, x):
        se = self.gap(x)
        se = self.dense1(se)
        se = self.dense2(se)
        se = self.reshape(se)
        return x * se

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "ratio": self.ratio})
        return config


class StochasticDepth(layers.Layer):
    """Drop path regularization for transformer blocks."""
    def __init__(self, drop_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.drop_rate = drop_rate

    def call(self, x, training=False):
        if not training or self.drop_rate == 0.0:
            return x
        keep_prob = 1 - self.drop_rate
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = tf.random.uniform(shape, dtype=x.dtype)
        binary_mask = tf.cast(random_tensor < keep_prob, x.dtype)
        return x * binary_mask / keep_prob

    def get_config(self):
        config = super().get_config()
        config.update({"drop_rate": self.drop_rate})
        return config


class TransformerBlock(layers.Layer):
    """Enhanced Transformer block with pre-norm and stochastic depth."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, drop_path_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.drop_path_rate = drop_path_rate

    def build(self, input_shape):
        self.att = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(self.ff_dim, activation="gelu"),
            layers.Dropout(self.rate),
            layers.Dense(self.embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.rate)
        self.dropout2 = layers.Dropout(self.rate)
        self.drop_path = StochasticDepth(self.drop_path_rate)
        super().build(input_shape)

    def call(self, inputs, training=False):
        # Pre-norm architecture
        x_norm = self.layernorm1(inputs)
        attn_output = self.att(x_norm, x_norm)
        attn_output = self.dropout1(attn_output, training=training)
        attn_output = self.drop_path(attn_output, training=training)
        out1 = inputs + attn_output

        x_norm2 = self.layernorm2(out1)
        ffn_output = self.ffn(x_norm2)
        ffn_output = self.dropout2(ffn_output, training=training)
        ffn_output = self.drop_path(ffn_output, training=training)
        return out1 + ffn_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
            "drop_path_rate": self.drop_path_rate,
        })
        return config


class PatchEmbedding(layers.Layer):
    """Converts CNN feature maps into patch embeddings with positional encoding."""
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        num_patches = input_shape[1]
        self.projection = layers.Dense(self.embed_dim)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.position_embedding = self.add_weight(
            name="pos_embed",
            shape=(1, num_patches, self.embed_dim),
            initializer="truncated_normal"
        )
        super().build(input_shape)

    def call(self, x):
        x = self.projection(x)
        x = x + self.position_embedding
        x = self.layer_norm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Warmup + cosine decay learning rate schedule."""
    def __init__(self, initial_lr=1e-4, peak_lr=1e-3, warmup_steps=500,
                 decay_steps=5000, alpha=1e-6):
        super().__init__()
        self.initial_lr = initial_lr
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.alpha = alpha

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.minimum(step / tf.maximum(tf.cast(self.warmup_steps, tf.float32), 1.0), 1.0)
        warmup_lr = self.initial_lr + (self.peak_lr - self.initial_lr) * warmup

        decay_step = tf.maximum(step - tf.cast(self.warmup_steps, tf.float32), 0.0)
        decay_steps_f = tf.cast(self.decay_steps, tf.float32)
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * tf.minimum(decay_step / decay_steps_f, 1.0)))
        cosine_lr = self.alpha + (self.peak_lr - self.alpha) * cosine_decay

        return tf.where(step < tf.cast(self.warmup_steps, tf.float32), warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
        }


def _residual_block(x, filters, use_se=True):
    """Residual CNN block with optional Squeeze-Excitation."""
    shortcut = x
    # Match channels for skip connection
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    if use_se:
        x = SqueezeExcitation(filters)(x)

    x = layers.Add()([shortcut, x])
    x = layers.Activation('gelu')(x)
    return x


def create_cnn_vit_model(input_shape=(128, 128, 1), num_classes=12):
    """
    Enhanced CNN + Vision Transformer hybrid model with:
    - 5 residual CNN blocks with SE attention
    - Multi-scale feature fusion from blocks 3/4/5
    - 4 Transformer blocks with stochastic depth
    - Wide classification head with GELU + LayerNorm
    - AdamW + warmup cosine schedule + label smoothing
    """
    inputs = layers.Input(shape=input_shape)

    # ---- Residual CNN Backbone ----
    # Block 1: 128x128 -> 64x64
    x = _residual_block(inputs, 32)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)

    # Block 2: 64x64 -> 32x32
    x = _residual_block(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)

    # Block 3: 32x32 -> 16x16 (multi-scale tap)
    x = _residual_block(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.15)(x)
    feat_16 = x  # 16x16x128

    # Block 4: 16x16 -> 8x8 (multi-scale tap)
    x = _residual_block(x, 256)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.15)(x)
    feat_8 = x  # 8x8x256

    # Block 5: 8x8 -> 4x4 (multi-scale tap)
    x = _residual_block(x, 512)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    feat_4 = x  # 4x4x512

    # ---- Multi-Scale Feature Fusion ----
    # Pool all features to 4x4, project to 128ch each, then concatenate
    f16 = layers.AveragePooling2D(pool_size=(4, 4))(feat_16)  # 16->4
    f16 = layers.Conv2D(128, (1, 1), activation='gelu')(f16)

    f8 = layers.AveragePooling2D(pool_size=(2, 2))(feat_8)  # 8->4
    f8 = layers.Conv2D(128, (1, 1), activation='gelu')(f8)

    f4 = layers.Conv2D(128, (1, 1), activation='gelu')(feat_4)

    fused = layers.Concatenate()([f16, f8, f4])  # 4x4x384

    # ---- Reshape to sequence for Transformer ----
    embed_dim = 192
    fused = layers.Reshape((16, 384))(fused)  # 4*4=16 patches, 384 channels

    # ---- Patch Embedding with positional encoding ----
    x = PatchEmbedding(embed_dim)(fused)

    # ---- Transformer Blocks with linearly increasing stochastic depth ----
    num_heads = 6
    ff_dim = 384
    num_transformer_blocks = 4
    max_drop_path = 0.15

    for i in range(num_transformer_blocks):
        drop_path_rate = max_drop_path * i / max(num_transformer_blocks - 1, 1)
        x = TransformerBlock(
            embed_dim, num_heads, ff_dim,
            rate=0.15, drop_path_rate=drop_path_rate
        )(x)

    # ---- Classification Head ----
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Activation('gelu')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(256)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Activation('gelu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Activation('gelu')(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    # AdamW with warmup + cosine decay
    lr_schedule = WarmupCosineDecay(
        initial_lr=1e-4, peak_lr=1e-3,
        warmup_steps=500, decay_steps=5000, alpha=1e-6
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=1e-4
    )

    # Label smoothing for better generalization (using CategoricalCrossentropy)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model


# ═══════════════════════════════════════════════════════════════
# SHAPES LIST — Only creative/non-geometric shapes (12 classes)
# ═══════════════════════════════════════════════════════════════
SHAPES = [
    "Spiral", "Infinity", "Cloud", "Lightning bolt", "Flower",
    "Butterfly", "Crown", "Flame", "Fish", "Leaf",
    "Music note", "Smiley face"
]

MODEL_PATH = 'shape_model_v2.h5'

# Training canvas size — matches webcam-like resolution so that stroke
# proportions and preprocessing behaviour are identical to inference.
DRAW_CANVAS = 480


# ═══════════════════════════════════════════════════════════════
# Data Augmentation Utilities (Enhanced)
# ═══════════════════════════════════════════════════════════════

def _bezier_curve(points, num_points=50):
    """Generate smooth bezier curve through control points."""
    points = np.array(points, dtype=np.float64)
    n = len(points) - 1
    t_values = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2))
    for i, t in enumerate(t_values):
        point = np.zeros(2)
        for j in range(n + 1):
            # Bernstein polynomial
            binom = math.factorial(n) / (math.factorial(j) * math.factorial(n - j))
            point += binom * (t ** j) * ((1 - t) ** (n - j)) * points[j]
        curve[i] = point
    return curve.astype(np.int32)


def _add_wobble(points, intensity=2.0):
    """Add random perturbations to simulate hand tremor."""
    pts = np.array(points, dtype=np.float64)
    noise = np.random.normal(0, intensity, pts.shape)
    return (pts + noise).astype(np.int32)


def _draw_varying_thickness(img, points, base_thickness=3, mode='taper'):
    """Draw stroke with varying thickness along the path."""
    n = len(points)
    if n < 2:
        return img
    for i in range(n - 1):
        t = i / max(n - 1, 1)
        if mode == 'taper':
            thick = max(1, int(base_thickness * (1 - 0.6 * t)))
        elif mode == 'reverse_taper':
            thick = max(1, int(base_thickness * (0.4 + 0.6 * t)))
        else:  # pulse
            thick = max(1, int(base_thickness * (0.7 + 0.3 * np.sin(t * np.pi * 3))))
        pt1 = (int(np.clip(points[i][0], 0, img.shape[1] - 1)),
               int(np.clip(points[i][1], 0, img.shape[0] - 1)))
        pt2 = (int(np.clip(points[i + 1][0], 0, img.shape[1] - 1)),
               int(np.clip(points[i + 1][1], 0, img.shape[0] - 1)))
        cv2.line(img, pt1, pt2, 255, thick)
    return img


def _elastic_deformation(img, alpha=15, sigma=3):
    """Elastic deformation to simulate hand tremor."""
    shape = img.shape
    dx = cv2.GaussianBlur(np.random.uniform(-1, 1, shape).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(np.random.uniform(-1, 1, shape).astype(np.float32), (0, 0), sigma) * alpha
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)


def _perspective_transform(img):
    """Apply slight random perspective warp."""
    size = img.shape[0]
    margin = int(size * 0.08)
    src = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    dst = np.float32([
        [np.random.randint(0, margin), np.random.randint(0, margin)],
        [size - np.random.randint(0, margin), np.random.randint(0, margin)],
        [size - np.random.randint(0, margin), size - np.random.randint(0, margin)],
        [np.random.randint(0, margin), size - np.random.randint(0, margin)]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (size, size))


def _cutout(img, num_holes=1, max_size=20):
    """Random rectangular erasing for occlusion simulation."""
    size = img.shape[0]
    for _ in range(num_holes):
        cx = np.random.randint(0, size)
        cy = np.random.randint(0, size)
        hw = np.random.randint(5, max_size)
        hh = np.random.randint(5, max_size)
        x1 = max(0, cx - hw)
        x2 = min(size, cx + hw)
        y1 = max(0, cy - hh)
        y2 = min(size, cy + hh)
        img[y1:y2, x1:x2] = 0
    return img


def augment_image(img):
    """Apply diverse random augmentations to a single image."""
    size = img.shape[0]

    # Random rotation (-30 to +30 degrees)
    angle = np.random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((size // 2, size // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (size, size))

    # Random scaling (0.8 to 1.2)
    scale = np.random.uniform(0.8, 1.2)
    M_scale = cv2.getRotationMatrix2D((size // 2, size // 2), 0, scale)
    img = cv2.warpAffine(img, M_scale, (size, size))

    # Random shearing (±10 degrees)
    if np.random.random() < 0.4:
        shear = np.random.uniform(-0.17, 0.17)  # ~10 degrees
        M_shear = np.float32([[1, shear, 0], [0, 1, 0]])
        img = cv2.warpAffine(img, M_shear, (size, size))

    # Random translation (-10 to +10 pixels)
    tx = np.random.randint(-10, 11)
    ty = np.random.randint(-10, 11)
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M_trans, (size, size))

    # Elastic deformation (40% chance)
    if np.random.random() < 0.4:
        img = _elastic_deformation(img, alpha=np.random.uniform(8, 20), sigma=np.random.uniform(2, 4))

    # Perspective transform (30% chance)
    if np.random.random() < 0.3:
        img = _perspective_transform(img)

    # Gaussian blur (30% chance)
    if np.random.random() < 0.3:
        ksize = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), np.random.uniform(0.5, 1.5))

    # Random noise
    noise_level = np.random.randint(0, 40)
    noise = np.random.randint(0, noise_level + 1, (size, size), dtype=np.uint8)
    img = cv2.add(img, noise)

    # Brightness/contrast variation (40% chance)
    if np.random.random() < 0.4:
        alpha_c = np.random.uniform(0.7, 1.3)
        beta_c = np.random.randint(-20, 21)
        img = np.clip(alpha_c * img.astype(np.float32) + beta_c, 0, 255).astype(np.uint8)

    # Morphological thickness variation
    if np.random.random() > 0.5:
        ksize = np.random.choice([2, 3])
        kernel = np.ones((ksize, ksize), np.uint8)
        if np.random.random() > 0.5:
            img = cv2.dilate(img, kernel, iterations=1)
        else:
            img = cv2.erode(img, kernel, iterations=1)

    # Cutout/random erasing (20% chance)
    if np.random.random() < 0.2:
        img = _cutout(img, num_holes=np.random.randint(1, 3), max_size=15)

    return img


def draw_shape_on_canvas(shape_name, size=128):
    """Draw a shape on a large canvas, then preprocess identically to inference."""
    img = np.zeros((DRAW_CANVAS, DRAW_CANVAS), dtype=np.uint8)

    # Parameters scaled for 480x480 canvas (simulates webcam-like drawing)
    center = (np.random.randint(140, 340), np.random.randint(140, 340))
    w = np.random.randint(40, 130)
    h = np.random.randint(40, 130)
    color = 255
    thickness = np.random.randint(2, 7)
    use_varying_thickness = np.random.random() < 0.3
    thickness_mode = np.random.choice(['taper', 'reverse_taper', 'pulse'])
    wobble_intensity = np.random.uniform(1.0, 5.0)
    is_partial = np.random.random() < 0.1

    if shape_name == "Spiral":
        variant = np.random.choice(["outward", "inward", "tight"])
        clockwise = np.random.choice([1, -1])
        turns = np.random.uniform(2.5, 5.5) if variant != "tight" else np.random.uniform(5, 8)
        pts = []
        step = 0.06 if variant == "tight" else 0.08
        for theta in np.arange(0, turns * np.pi, step):
            r = (theta / (turns * np.pi)) * w
            if variant == "inward":
                r = w - r
            px = int(center[0] + r * np.cos(clockwise * theta))
            py = int(center[1] + r * np.sin(clockwise * theta))
            pts.append([px, py])
        if is_partial:
            pts = pts[:int(len(pts) * np.random.uniform(0.5, 0.8))]
        pts = _add_wobble(pts, wobble_intensity)
        if len(pts) > 1:
            if use_varying_thickness:
                _draw_varying_thickness(img, pts, thickness, thickness_mode)
            else:
                cv2.polylines(img, [np.array(pts)], False, color, thickness)

    elif shape_name == "Infinity":
        variant = np.random.choice(["standard", "wide", "tall"])
        pts = []
        scale_x = np.random.uniform(1.2, 1.8)
        scale_y = 1.0 if variant != "tall" else np.random.uniform(1.2, 1.5)
        if variant == "wide":
            scale_x *= 1.3
        for t in np.arange(0, 2 * np.pi, 0.06):
            den = 1 + np.sin(t) ** 2
            px = int(center[0] + w * scale_x * np.cos(t) / den)
            py = int(center[1] + w * scale_y * np.sin(t) * np.cos(t) / den)
            pts.append([px, py])
        pts = _add_wobble(pts, wobble_intensity)
        if len(pts) > 1:
            if use_varying_thickness:
                _draw_varying_thickness(img, pts, thickness, thickness_mode)
            else:
                cv2.polylines(img, [np.array(pts)], True, color, thickness)

    elif shape_name == "Cloud":
        # Cloud: flat/gently-curved bottom + bumpy semicircular top (NOT radial sinusoid)
        num_bumps = np.random.randint(3, 6)
        cloud_w = w * np.random.uniform(0.8, 1.2)
        cloud_h = h * np.random.uniform(0.5, 0.8)
        bump_r = cloud_h * np.random.uniform(0.3, 0.5)
        pts = []
        # Bottom: gentle flat line from left to right
        bottom_y = center[1] + int(cloud_h * 0.3)
        sag = np.random.uniform(0, cloud_h * 0.15)
        for i in range(20):
            t = i / 19.0
            bx = int(center[0] - cloud_w + 2 * cloud_w * t)
            by = int(bottom_y + sag * np.sin(t * np.pi))
            pts.append([bx, by])
        # Top: semicircular bumps from right to left
        bump_centers = np.linspace(center[0] + cloud_w * 0.8, center[0] - cloud_w * 0.8, num_bumps)
        top_y = center[1] - int(cloud_h * 0.3)
        for bc in bump_centers:
            br = bump_r * np.random.uniform(0.7, 1.3)
            for a in np.arange(0, np.pi + 0.1, 0.15):
                bx = int(bc + br * np.cos(a + np.pi))
                by = int(top_y - br * np.sin(a))
                pts.append([bx, by])
        pts.append(pts[0])  # close the shape
        if is_partial:
            pts = pts[:int(len(pts) * np.random.uniform(0.6, 0.85))]
        pts = _add_wobble(pts, wobble_intensity)
        if len(pts) > 1:
            if use_varying_thickness:
                _draw_varying_thickness(img, pts, thickness, thickness_mode)
            else:
                cv2.polylines(img, [np.array(pts)], False, color, thickness)

    elif shape_name == "Lightning bolt":
        variant = np.random.choice(["zigzag", "curved", "forked"])
        num_zags = np.random.randint(3, 7)
        pts = [[center[0], center[1] - h]]
        for i in range(1, num_zags):
            direction = 1 if i % 2 == 0 else -1
            px = center[0] + direction * np.random.randint(w // 4, w // 2 + 1)
            py = center[1] - h + int((2 * h / num_zags) * i)
            pts.append([px, py])
        pts.append([center[0], center[1] + h])
        if variant == "curved":
            if len(pts) >= 4:
                pts = _bezier_curve(pts, num_points=40).tolist()
        pts = _add_wobble(pts, wobble_intensity)
        if len(pts) > 1:
            if use_varying_thickness:
                _draw_varying_thickness(img, pts, thickness, thickness_mode)
            else:
                cv2.polylines(img, [np.array(pts)], False, color, thickness)
        if variant == "forked" and len(pts) > 3:
            fork_pt = pts[len(pts) // 2]
            fork_end = [fork_pt[0] + np.random.randint(10, 25), fork_pt[1] + np.random.randint(10, 25)]
            cv2.line(img, tuple(fork_pt), tuple(fork_end), color, max(1, thickness - 1))

    elif shape_name == "Flower":
        num_petals = np.random.randint(4, 8)
        center_r = np.random.randint(max(3, w // 6), max(5, w // 4))
        petal_len = np.random.uniform(0.4, 0.7) * w
        pts = []
        phase = np.random.uniform(0, 2 * np.pi)
        for t in np.arange(0, 2 * np.pi + 0.1, 0.04):
            petal = abs(np.sin(num_petals * t / 2 + phase))
            r = center_r + petal_len * petal
            px = int(center[0] + r * np.cos(t))
            py = int(center[1] + r * np.sin(t))
            pts.append([px, py])
        if is_partial:
            pts = pts[:int(len(pts) * np.random.uniform(0.6, 0.85))]
        pts = _add_wobble(pts, wobble_intensity)
        if len(pts) > 1:
            if use_varying_thickness:
                _draw_varying_thickness(img, pts, thickness, thickness_mode)
            else:
                cv2.polylines(img, [np.array(pts)], True, color, thickness)

    elif shape_name == "Butterfly":
        # Butterfly: two mirrored wing outlines that meet at center body line
        wing_w = w * np.random.uniform(0.7, 1.2)
        upper_h = h * np.random.uniform(0.6, 0.9)
        lower_h = h * np.random.uniform(0.3, 0.5)
        pts = []
        # Body line (vertical center)
        body_top = center[1] - int(upper_h * 0.8)
        body_bot = center[1] + int(lower_h * 0.8)
        # Left upper wing (elliptical arc going left then back to center)
        for t in np.arange(0, np.pi + 0.1, 0.08):
            wx = int(center[0] - wing_w * np.sin(t))
            wy = int(center[1] - upper_h * 0.6 * np.sin(t) - upper_h * 0.2 * (1 - np.cos(t)))
            pts.append([wx, wy])
        # Left lower wing (smaller, going left-down then back)
        for t in np.arange(0, np.pi + 0.1, 0.1):
            wx = int(center[0] - wing_w * 0.7 * np.sin(t))
            wy = int(center[1] + lower_h * 0.7 * np.sin(t) + lower_h * 0.1 * (1 - np.cos(t)))
            pts.append([wx, wy])
        # Right lower wing (mirrored)
        for t in np.arange(0, np.pi + 0.1, 0.1):
            wx = int(center[0] + wing_w * 0.7 * np.sin(t))
            wy = int(center[1] + lower_h * 0.7 * np.sin(t) + lower_h * 0.1 * (1 - np.cos(t)))
            pts.append([wx, wy])
        # Right upper wing (mirrored)
        for t in np.arange(0, np.pi + 0.1, 0.08):
            wx = int(center[0] + wing_w * np.sin(t))
            wy = int(center[1] - upper_h * 0.6 * np.sin(t) - upper_h * 0.2 * (1 - np.cos(t)))
            pts.append([wx, wy])
        if is_partial:
            pts = pts[:int(len(pts) * np.random.uniform(0.6, 0.85))]
        pts = _add_wobble(pts, wobble_intensity)
        if len(pts) > 1:
            if use_varying_thickness:
                _draw_varying_thickness(img, pts, thickness, thickness_mode)
            else:
                cv2.polylines(img, [np.array(pts)], False, color, thickness)
        # Body line through center
        body_pts = _add_wobble([[center[0], body_top],
                                [center[0], body_bot]], wobble_intensity * 0.5)
        cv2.polylines(img, [np.array(body_pts)], False, color, max(1, thickness - 1))

    elif shape_name == "Crown":
        num_peaks = np.random.randint(3, 6)
        pts = [[center[0] - w, center[1] + h // 2]]
        for i in range(num_peaks):
            frac = (i + 0.5) / num_peaks
            tip_x = int(center[0] - w + 2 * w * frac)
            tip_y = center[1] - h // 2 + np.random.randint(-7, 7)
            valley_x = int(center[0] - w + 2 * w * (i + 1) / num_peaks)
            valley_y = center[1] + np.random.randint(-5, 7)
            pts.append([tip_x, tip_y])
            if i < num_peaks - 1:
                pts.append([valley_x, valley_y])
        pts.append([center[0] + w, center[1] + h // 2])
        pts = _add_wobble(pts, wobble_intensity)
        cv2.polylines(img, [np.array(pts)], True, color, thickness)

    elif shape_name == "Flame":
        # Flame: teardrop shape — wide base, pointed top (like a candle flame)
        flame_w = w * np.random.uniform(0.6, 1.0)
        flame_h = h * np.random.uniform(1.0, 1.5)
        taper = np.random.uniform(1.5, 3.0)  # how pointy the top is
        pts = []
        for t in np.arange(0, 2 * np.pi + 0.05, 0.06):
            # Teardrop parametric: wider at bottom (t=pi), pointed at top (t=0)
            # Use sin(t/2)^taper to create the taper effect
            r_x = flame_w * np.sin(t)
            r_y = flame_h * (max(0.0, np.sin(t / 2)) ** taper - 0.5)
            px = int(center[0] + r_x)
            py = int(center[1] - r_y)
            pts.append([px, py])
        pts = _add_wobble(pts, wobble_intensity)
        if len(pts) > 1:
            if use_varying_thickness:
                _draw_varying_thickness(img, pts, thickness, thickness_mode)
            else:
                cv2.polylines(img, [np.array(pts)], True, color, thickness)

    elif shape_name == "Fish":
        body_w = np.random.randint(max(10, w - 5), w + 5)
        body_h = np.random.randint(max(5, h // 2 - 3), h // 2 + 5)
        tail_w = np.random.randint(w // 3, w // 2 + 3)
        pts = []
        # Body as continuous ellipse starting from left, going up-right-down-left
        for t in np.linspace(np.pi, -np.pi, 60):
            px = int(center[0] + body_w * np.cos(t))
            py = int(center[1] - body_h * np.sin(t))
            pts.append([px, py])
        # Tail V-shape: from left of body out and back
        pts.append([center[0] - body_w, center[1]])
        pts.append([center[0] - body_w - tail_w, center[1] - body_h])
        pts.append([center[0] - body_w, center[1]])
        pts.append([center[0] - body_w - tail_w, center[1] + body_h])
        pts.append([center[0] - body_w, center[1]])
        if is_partial:
            pts = pts[:int(len(pts) * np.random.uniform(0.6, 0.85))]
        pts = _add_wobble(pts, wobble_intensity)
        if len(pts) > 1:
            if use_varying_thickness:
                _draw_varying_thickness(img, pts, thickness, thickness_mode)
            else:
                cv2.polylines(img, [np.array(pts)], False, color, thickness)

    elif shape_name == "Leaf":
        pts_upper = []
        pts_lower = []
        leaf_w_var = np.random.uniform(0.7, 1.3)
        for t in np.linspace(0, np.pi, 30):
            pts_upper.append([int(center[0] - w + 2 * w * t / np.pi),
                              int(center[1] - h * leaf_w_var * np.sin(t))])
        for t in np.linspace(0, np.pi, 30):
            pts_lower.append([int(center[0] + w - 2 * w * t / np.pi),
                              int(center[1] + h * leaf_w_var * np.sin(t))])
        all_pts = pts_upper + pts_lower
        all_pts = _add_wobble(all_pts, wobble_intensity)
        if len(all_pts) > 1:
            cv2.polylines(img, [np.array(all_pts)], True, color, thickness)
        cv2.line(img, (center[0] - w, center[1]), (center[0] + w, center[1]),
                 color, max(1, thickness - 1))

    elif shape_name == "Music note":
        variant = np.random.choice(["single", "beamed", "double"])
        head_r = np.random.randint(max(3, w // 5), max(5, w // 3))
        pts = []
        if variant == "single":
            head_x = center[0] - w // 3
            head_y = center[1] + h // 3
            # Note head as continuous oval
            for t in np.arange(0, 2 * np.pi + 0.1, 0.15):
                px = int(head_x + head_r * 1.2 * np.cos(t))
                py = int(head_y + head_r * 0.7 * np.sin(t))
                pts.append([px, py])
            # Stem going up from right of head
            stem_x = head_x + head_r
            stem_top = center[1] - h
            for yv in np.linspace(head_y, stem_top, 20):
                pts.append([stem_x, int(yv)])
            # Flag curving right and down
            for t in np.linspace(0, 1, 15):
                fx = int(stem_x + (w // 3) * np.sin(t * np.pi * 0.6))
                fy = int(stem_top + (head_y - stem_top) * 0.4 * t)
                pts.append([fx, fy])
        elif variant == "beamed":
            head_x1 = center[0] - w // 2
            head_x2 = center[0] + w // 4
            head_y = center[1] + h // 3
            stem_top = center[1] - h
            # First note head
            for t in np.arange(0, 2 * np.pi + 0.1, 0.15):
                pts.append([int(head_x1 + head_r * 1.2 * np.cos(t)),
                            int(head_y + head_r * 0.7 * np.sin(t))])
            # First stem up
            stem_x1 = head_x1 + head_r
            for yv in np.linspace(head_y, stem_top, 15):
                pts.append([stem_x1, int(yv)])
            # Beam across to second stem
            stem_x2 = head_x2 + head_r
            for xv in np.linspace(stem_x1, stem_x2, 10):
                pts.append([int(xv), stem_top])
            # Second stem down
            for yv in np.linspace(stem_top, head_y, 15):
                pts.append([stem_x2, int(yv)])
            # Second note head
            for t in np.arange(0, 2 * np.pi + 0.1, 0.15):
                pts.append([int(head_x2 + head_r * 1.2 * np.cos(t)),
                            int(head_y + head_r * 0.7 * np.sin(t))])
        else:  # double flag
            head_x = center[0] - w // 3
            head_y = center[1] + h // 3
            stem_top = center[1] - h
            # Note head
            for t in np.arange(0, 2 * np.pi + 0.1, 0.15):
                pts.append([int(head_x + head_r * 1.2 * np.cos(t)),
                            int(head_y + head_r * 0.7 * np.sin(t))])
            # Stem up
            stem_x = head_x + head_r
            for yv in np.linspace(head_y, stem_top, 20):
                pts.append([stem_x, int(yv)])
            # First flag
            for t in np.linspace(0, 1, 12):
                fx = int(stem_x + (w // 3) * np.sin(t * np.pi * 0.6))
                fy = int(stem_top + (head_y - stem_top) * 0.35 * t)
                pts.append([fx, fy])
            # Back to stem for second flag
            pts.append([stem_x, int(stem_top + (head_y - stem_top) * 0.3)])
            for t in np.linspace(0, 1, 12):
                fx = int(stem_x + (w // 3) * np.sin(t * np.pi * 0.6))
                fy = int(stem_top + (head_y - stem_top) * (0.3 + 0.35 * t))
                pts.append([fx, fy])
        pts = _add_wobble(pts, wobble_intensity)
        if len(pts) > 1:
            if use_varying_thickness:
                _draw_varying_thickness(img, pts, thickness, thickness_mode)
            else:
                cv2.polylines(img, [np.array(pts)], False, color, thickness)

    elif shape_name == "Smiley face":
        face_r = np.random.randint(max(10, w - 5), w + 5)
        pts = []
        # Face circle as continuous path
        for t in np.arange(0, 2 * np.pi + 0.1, 0.06):
            px = int(center[0] + face_r * np.cos(t))
            py = int(center[1] + face_r * np.sin(t))
            pts.append([px, py])
        # Move to left eye and draw small circle
        eye_r = max(3, face_r // 6)
        eye_y = center[1] - face_r // 3
        eye_offset = face_r // 3
        left_eye = (center[0] - eye_offset, eye_y)
        pts.append(list(left_eye))
        for t in np.arange(0, 2 * np.pi + 0.1, 0.3):
            pts.append([int(left_eye[0] + eye_r * np.cos(t)),
                        int(left_eye[1] + eye_r * np.sin(t))])
        # Move to right eye and draw small circle
        right_eye = (center[0] + eye_offset, eye_y)
        pts.append(list(right_eye))
        for t in np.arange(0, 2 * np.pi + 0.1, 0.3):
            pts.append([int(right_eye[0] + eye_r * np.cos(t)),
                        int(right_eye[1] + eye_r * np.sin(t))])
        # Move to smile and draw arc
        smile_w = face_r // 2
        smile_h = np.random.randint(max(2, face_r // 5), max(4, face_r // 3))
        smile_cy = center[1] + face_r // 5
        pts.append([center[0] - smile_w, smile_cy])
        for t in np.arange(0, np.pi + 0.1, 0.1):
            sx = int(center[0] - smile_w + 2 * smile_w * t / np.pi)
            sy = int(smile_cy + smile_h * np.sin(t))
            pts.append([sx, sy])
        pts = _add_wobble(pts, wobble_intensity)
        if len(pts) > 1:
            if use_varying_thickness:
                _draw_varying_thickness(img, pts, thickness, thickness_mode)
            else:
                cv2.polylines(img, [np.array(pts)], False, color, thickness)

    # CRITICAL: apply the same crop→pad→resize as inference pipeline
    processed = preprocess_canvas(img, output_size=size)
    return processed if processed is not None else np.zeros((size, size), dtype=np.uint8)


def generate_synthetic_data(samples_per_class=2500):
    """
    Generates synthetic shape data with heavy augmentation.
    Uses 95% augmentation rate, intra-class mixup, and hard negative awareness.
    """
    X, y = [], []
    size = 128

    total = len(SHAPES) * samples_per_class
    print(f"Generating {total} samples ({samples_per_class} per class x {len(SHAPES)} classes)...")

    for idx, shape_name in enumerate(SHAPES):
        print(f"  [{idx+1}/{len(SHAPES)}] Generating '{shape_name}'...")
        class_images = []
        for sample_i in range(samples_per_class):
            img = draw_shape_on_canvas(shape_name, size)

            # Apply augmentation to 95% of samples
            if np.random.random() < 0.95:
                img = augment_image(img)
            else:
                noise = np.random.randint(0, 20, (size, size), dtype=np.uint8)
                img = cv2.add(img, noise)

            img = img.astype('float32') / 255.0
            class_images.append(img)
            X.append(np.expand_dims(img, axis=-1))
            y.append(idx)

        # Mixup: one image dominates, second adds faint variation (not 50/50 ghost)
        num_mixup = max(1, samples_per_class // 10)
        for _ in range(num_mixup):
            i1, i2 = np.random.choice(len(class_images), 2, replace=False)
            lam = np.random.uniform(0.85, 0.95)
            mixed = lam * class_images[i1] + (1 - lam) * class_images[i2]
            X.append(np.expand_dims(mixed, axis=-1))
            y.append(idx)

    X = np.array(X, dtype='float32')
    y = np.array(y)

    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    print(f"  Total samples (with mixup): {len(X)}")
    return X, y


def train_model():
    """
    Three-phase training strategy for maximum accuracy.
    GPU memory growth enabled for small-VRAM GPUs (4GB RTX 2050).
    Phase 1: Initial training with moderate data for convergence
    Phase 2: Fine-tune with more data and lower LR
    Phase 3: Final polish with heavy augmentation and very low LR
    """
    # Enable GPU memory growth to avoid OOM on small-VRAM GPUs
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("=" * 60)
    print("  TRAINING ENHANCED CNN + VISION TRANSFORMER MODEL")
    print(f"  GPUs: {len(gpus)} | Classes: {SHAPES}")
    print("=" * 60)

    BATCH_SIZE = 16 if gpus and len(gpus) > 0 else 32

    # ---- Phase 1: Initial Training ----
    print("\n" + "=" * 60)
    print("  PHASE 1: Initial Training (600 samples/class)")
    print("=" * 60)

    num_classes = len(SHAPES)

    print("\n[1/8] Generating Phase 1 training data...")
    X1, y1_sparse = generate_synthetic_data(600)
    y1 = tf.keras.utils.to_categorical(y1_sparse, num_classes)
    print(f"  Data shape: X={X1.shape}, y={y1.shape}")

    print("\n[2/8] Creating CNN+ViT model...")
    model = create_cnn_vit_model(num_classes=num_classes)
    model.summary()

    print("\n[3/8] Phase 1 training...")
    callbacks_p1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=20,
            restore_best_weights=True, min_delta=0.001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
    ]

    history1 = model.fit(
        X1, y1,
        epochs=40,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=callbacks_p1,
        verbose=1
    )

    val_acc1 = max(history1.history.get('val_accuracy', [0]))
    print(f"\n  Phase 1 Best Val Accuracy: {val_acc1:.4f}")

    # Free memory
    del X1, y1, y1_sparse

    # ---- Phase 2: Fine-tuning with more data ----
    print("\n" + "=" * 60)
    print("  PHASE 2: Fine-tuning (1000 samples/class)")
    print("=" * 60)

    print("\n[4/8] Generating Phase 2 training data...")
    X2, y2_sparse = generate_synthetic_data(1000)
    y2 = tf.keras.utils.to_categorical(y2_sparse, num_classes)
    print(f"  Data shape: X={X2.shape}, y={y2.shape}")

    # Recompile with lower learning rate for fine-tuning
    lr_schedule_ft = WarmupCosineDecay(
        initial_lr=1e-5, peak_lr=3e-4,
        warmup_steps=300, decay_steps=4000, alpha=1e-7
    )
    optimizer_ft = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule_ft, weight_decay=5e-5
    )
    model.compile(
        optimizer=optimizer_ft,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )

    print("\n[5/8] Phase 2 fine-tuning...")
    callbacks_p2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=20,
            restore_best_weights=True, min_delta=0.0005
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
    ]

    history2 = model.fit(
        X2, y2,
        epochs=30,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=callbacks_p2,
        verbose=1
    )

    val_acc2 = max(history2.history.get('val_accuracy', [0]))
    print(f"\n  Phase 2 Best Val Accuracy: {val_acc2:.4f}")

    # Free memory
    del X2, y2, y2_sparse

    # ---- Phase 3: Final polish with heavy augmentation ----
    print("\n" + "=" * 60)
    print("  PHASE 3: Final Polish (1200 samples/class, heavy aug)")
    print("=" * 60)

    print("\n[6/8] Generating Phase 3 training data...")
    X3, y3_sparse = generate_synthetic_data(1200)
    y3 = tf.keras.utils.to_categorical(y3_sparse, num_classes)
    print(f"  Data shape: X={X3.shape}, y={y3.shape}")

    # Very low learning rate for final refinement
    lr_schedule_p3 = WarmupCosineDecay(
        initial_lr=1e-6, peak_lr=1e-4,
        warmup_steps=200, decay_steps=3000, alpha=1e-8
    )
    optimizer_p3 = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule_p3, weight_decay=3e-5
    )
    model.compile(
        optimizer=optimizer_p3,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.03),
        metrics=['accuracy']
    )

    print("\n[7/8] Phase 3 final polish...")
    callbacks_p3 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=15,
            restore_best_weights=True, min_delta=0.0003
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
    ]

    history3 = model.fit(
        X3, y3,
        epochs=25,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=callbacks_p3,
        verbose=1
    )

    val_acc3 = max(history3.history.get('val_accuracy', [0]))
    train_acc3 = max(history3.history.get('accuracy', [0]))

    # ---- Evaluation: Per-class accuracy ----
    print("\n[8/8] Evaluating per-class accuracy...")
    split_idx = int(len(X3) * 0.85)
    X_val = X3[split_idx:]
    y_val_sparse = y3_sparse[split_idx:]

    preds = model.predict(X_val, verbose=0)
    pred_classes = np.argmax(preds, axis=1)

    print(f"\n{'=' * 60}")
    print(f"  PER-CLASS ACCURACY")
    print(f"{'=' * 60}")
    confused_pairs = Counter()
    for idx, shape_name in enumerate(SHAPES):
        mask = y_val_sparse == idx
        if mask.sum() > 0:
            correct = (pred_classes[mask] == idx).sum()
            total = mask.sum()
            acc = correct / total
            print(f"  {shape_name:20s}: {acc:.4f} ({correct}/{total})")
            # Track confusion
            wrong_mask = mask & (pred_classes != idx)
            if wrong_mask.sum() > 0:
                wrong_preds = pred_classes[wrong_mask]
                for wp in wrong_preds:
                    confused_pairs[(shape_name, SHAPES[wp])] += 1

    if confused_pairs:
        print(f"\n  Top confused pairs:")
        for (a, b), count in confused_pairs.most_common(10):
            print(f"    {a} -> {b}: {count}")

    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Phase 1 Best Val Accuracy: {val_acc1:.4f}")
    print(f"  Phase 2 Best Val Accuracy: {val_acc2:.4f}")
    print(f"  Phase 3 Best Val Accuracy: {val_acc3:.4f}")
    print(f"  Phase 3 Best Train Accuracy: {train_acc3:.4f}")
    print(f"{'=' * 60}")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Free memory
    del X3, y3, y3_sparse

    return model


def load_model():
    """Load the trained model or train a new one if not found."""
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={
                    'TransformerBlock': TransformerBlock,
                    'PatchEmbedding': PatchEmbedding,
                    'SqueezeExcitation': SqueezeExcitation,
                    'StochasticDepth': StochasticDepth,
                    'WarmupCosineDecay': WarmupCosineDecay,
                }
            )
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Retraining...")
            return train_model()
    else:
        return train_model()
