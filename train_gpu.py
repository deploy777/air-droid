#!/usr/bin/env python3
"""GPU training launcher — sets up CUDA paths and runs training."""
import os, sys, glob

# Set CUDA library paths from pip-installed nvidia packages
nvidia_lib_dirs = glob.glob(os.path.expanduser(
    '~/.local/lib/python3.12/site-packages/nvidia/*/lib'))
if nvidia_lib_dirs:
    os.environ['LD_LIBRARY_PATH'] = ':'.join(nvidia_lib_dirs) + ':' + os.environ.get('LD_LIBRARY_PATH', '')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# Enable memory growth for small-VRAM GPUs
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(f"GPUs available: {len(gpus)}")

if not gpus:
    print("WARNING: No GPU detected! Training will be slow on CPU.")
    print("Make sure nvidia drivers + CUDA toolkit are installed.")

from model import train_model
train_model()
