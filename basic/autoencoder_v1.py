from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as imshow
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array, load_img
# Define the size of the input images

np.random.seed(42)  # For reproducibility

SIZE = 256

img_data = []

img = cv2.imread('images/monalisa.jpg')  # Load an image from the specified path
# Initialize the Sequential model
model = Sequential()

# Define the autoencoder architecture
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

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Display the model summary
model.summary()