import tensorflow as tf
import cv2
import pandas as pd
import numpy as np

print("TensorFlow version:", tf.__version__)
print("OpenCV version:", cv2.__version__)
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("\nAll packages imported successfully!")

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU available: {len(gpus)} GPU(s) found")
