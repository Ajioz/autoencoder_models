import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set random seed
np.random.seed(42)
SIZE = 256

# Load and prepare image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(BASE_DIR, 'images/monalisa.jpg')

img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {img_path}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (SIZE, SIZE))
img_array = np.reshape([img_to_array(img)], (1, SIZE, SIZE, 3)).astype('float32') / 255.0

# Build the autoencoder
model = Sequential([
    # Encoder
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(8, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(8, (3, 3), activation='relu', padding='same'),

    # Decoder
    UpSampling2D((2, 2)),
    Conv2D(8, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Conv2D(3, (3, 3), activation='sigmoid', padding='same')
])

# Show model summary
model.summary()

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Set callbacks
early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.h5", monitor='loss', save_best_only=True, mode='min', verbose=1)

# Redirect stdout and stderr to summary.txt
log_path = os.path.join(BASE_DIR, "summary.txt")
with open(log_path, "w") as f:
    sys.stdout = sys.stderr = f  # redirect both stdout and stderr to file

    # Display model summary in file
    model.summary()

    # Train and collect history
    history = model.fit(
        img_array, img_array,
        epochs=400,
        batch_size=1,
        shuffle=True,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

# Restore stdout/stderr after training
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# Predict and show results
predicted = model.predict(img_array)

# Plot original vs reconstructed
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img_array[0])
axs[0].set_title("Original")
axs[0].axis("off")

axs[1].imshow(predicted[0])
axs[1].set_title("Reconstructed")
axs[1].axis("off")
plt.tight_layout()
plt.show()

# Plot loss curve
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)
plt.show()
