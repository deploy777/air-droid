# AI Air Drawing — Creative Shape Detector

Draw shapes in the air using your index finger and let AI detect them in real-time. Uses a CNN + Vision Transformer model trained on 12 creative shapes.

## Supported Shapes

🌀 Spiral · ♾️ Infinity · ☁️ Cloud · ⚡ Lightning bolt · 🌸 Flower · 🦋 Butterfly · 👑 Crown · 🔥 Flame · 🐟 Fish · 🍃 Leaf · 🎵 Music note · 😊 Smiley face

## Quick Start

**Prerequisites:** Python 3.10+, a webcam, and [Git LFS](https://git-lfs.github.com/) installed.

```bash
# 1. Clone (Git LFS pulls the model automatically)
git lfs install
git clone https://github.com/AVISHKAR-PROJECTS-HACKNCRAFTS/AI-Air-Drawing-Shape-Detection.git
cd AI-Air-Drawing-Shape-Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

> **If the model fails to load**, run `git lfs pull` inside the repo to download the model file.

## How to Use

1. Click **Start Camera** in the sidebar
2. Hold up your **index finger** (keep other fingers closed) to draw
3. When you pull your hand away, the AI classifies the shape
4. Detected shapes appear on the canvas with confidence scores

## Architecture

- **Model:** 4-block CNN with Squeeze-and-Excitation attention + 3-layer Vision Transformer
- **Parameters:** ~1.68M
- **Input:** 128x128 grayscale images
- **Training:** 3-phase curriculum (600 → 1500 → 2500 samples/class) with heavy augmentation

## Re-training (Optional — Developers Only)

If you want to retrain the model with a GPU:

```bash
python train_gpu.py
```

This generates synthetic training data and trains the CNN+ViT model from scratch. Requires a CUDA-capable GPU for reasonable training times.
