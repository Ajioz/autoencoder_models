from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential

SIZE = 256

# Define layers in a list
layers = [
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(8, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(8, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    UpSampling2D((2, 2)),
    Conv2D(8, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    UpSampling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(3, (3, 3), activation='relu', padding='same'),
    Dropout(0.2)  # 20% dropout to prevent overfitting
]

# Build model using Sequential with layers list
model = Sequential(layers)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.summary()