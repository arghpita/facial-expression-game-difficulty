import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("Facial Expression Recognition Training")
print("=" * 60)

# Load and preprocess data
print("\n[1/6] Loading FER2013 dataset...")
df = pd.read_csv('fer2013.csv')
print(f"Dataset loaded: {len(df)} samples")
print(f"Columns: {df.columns.tolist()}")

# Expression labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
print(f"Emotion classes: {emotion_labels}")

# Prepare data
print("\n[2/6] Preprocessing data...")
X = []
y = []

for index, row in df.iterrows():
    pixels = np.array(row['pixels'].split(), dtype='float32')
    pixels = pixels.reshape(48, 48, 1)  # Reshape to 48x48x1
    pixels = pixels / 255.0  # Normalize to [0, 1]
    X.append(pixels)
    y.append(row['emotion'])
    
    if (index + 1) % 5000 == 0:
        print(f"  Processed {index + 1}/{len(df)} images...")

X = np.array(X)
y = np.array(y)

print(f"Data shape: X={X.shape}, y={y.shape}")

# Split data
print("\n[3/6] Splitting dataset...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")

# Build CNN model
print("\n[4/6] Building CNN model...")
model = models.Sequential([
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Third convolutional block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Flatten and dense layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model.summary()

# Train model
print("\n[5/6] Training model...")
print("This may take a while depending on your hardware...\n")

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train
history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# Evaluate model
print("\n[6/6] Evaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Save final model
model.save('facial_expression_model.keras')
print("\nModel saved as 'facial_expression_model.keras'")
print("Best model saved as 'best_model.keras'")

# Plot training history
print("\nGenerating training plots...")
plt.figure(figsize=(14, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("Training plots saved as 'training_history.png'")

# Print summary
print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Total Epochs Trained: {len(history.history['loss'])}")
print("\nFiles created:")
print("  - facial_expression_model.keras (final model)")
print("  - best_model.keras (best validation accuracy)")
print("  - training_history.png (training plots)")
print("=" * 60)
