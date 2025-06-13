import matplotlib.pyplot as imshow
import numpy as np
import cv2

from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

# Set random seed for reproducibility
np.random.seed(42)  

SIZE = 256

img_data = []

img = cv2.imread('images/monalisa.jpg', 1)  # Load an image from the specified path as colored hence --> 1
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB format
img = cv2.resize(img, (SIZE, SIZE))  # Resize the image to the specified size
img_data.append(img_to_array(img))  # Append the image to the list

img_array = np.reshape(img_data,(len(img_data), SIZE, SIZE, 3))  # Reshape the list to a numpy array
img_array = img_array.astype('float32') / 255.0  # Normalize the pixel values to the range [0, 1]




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

model.fit(img_array, img_array, epochs=5, batch_size=1, shuffle=True)

predicted = model.predict(img_array)

imshow.imshow(predicted[0].reshape(SIZE, SIZE, 3))  # Display the original image