import os
import matplotlib.pyplot as plt  # Changed alias from 'imshow' to 'plt'
import numpy as np
import cv2

from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

# Set random seed for reproducibility
np.random.seed(42)

SIZE = 256
img_data = []

# Load image using relative path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(BASE_DIR, 'images/monalisa.jpg')  # Adjust if needed

img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {img_path}")

# Preprocess image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (SIZE, SIZE))
img_data.append(img_to_array(img))

# Convert to numpy array and normalize
img_array = np.reshape(img_data, (len(img_data), SIZE, SIZE, 3))
img_array = img_array.astype('float32') / 255.0

# Define autoencoder model
model = Sequential()
# Encoder
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
# Decoder
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

# Compile and train
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.summary()
model.fit(img_array, img_array, epochs=400, batch_size=1, shuffle=True)

# Predict reconstructed image
predicted = model.predict(img_array)

# Display original and reconstructed images side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(img_array[0])
axs[0].set_title("Original")
axs[0].axis('off')

axs[1].imshow(predicted[0])
axs[1].set_title("Reconstructed")
axs[1].axis('off')

plt.tight_layout()
plt.show()
