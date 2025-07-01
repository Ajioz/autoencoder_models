import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
img_width, img_height = x_train.shape[1], x_train.shape[2]
num_channels = 1
x_train = x_train.reshape((-1, img_height, img_width, num_channels))
x_test = x_test.reshape((-1, img_height, img_width, num_channels))

input_shape = (img_height, img_width, num_channels)

# Encoder
latent_dim = 2
input_img = Input(shape=input_shape)
x = Conv2D(32, 3, padding='same', activation='relu')(input_img)
x = Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
conv_shape = K.int_shape(x)

x = Flatten()(x)
x = Dense(32, activation='relu')(x)

z_mu = Dense(latent_dim)(x)
z_sigma = Dense(latent_dim)(x)

# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, name='z')([z_mu, z_sigma])

encoder = Model(input_img, [z_mu, z_sigma, z], name="encoder")

# Decoder
decoder_input = Input(shape=(latent_dim,))
x = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu')(decoder_input)
x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
x = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = Conv2DTranspose(num_channels, 3, padding='same', activation='sigmoid')(x)

decoder = Model(decoder_input, x, name="decoder")
z_decoded = decoder(z)

# Define VAE loss
def vae_loss(x_true, x_pred, z_mu=z_mu, z_sigma=z_sigma):
    x_true = tf.keras.backend.flatten(x_true)
    x_pred = tf.keras.backend.flatten(x_pred)
    recon_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x_true, x_pred))
    kl_loss = -0.5 * tf.reduce_mean(1 + z_sigma - tf.square(z_mu) - tf.exp(z_sigma))
    return recon_loss + 5e-4 * kl_loss

# Wrap the loss in a Lambda layer
class VAELossLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        x_true = inputs[0]
        x_pred = inputs[1]
        loss = vae_loss(x_true, x_pred)
        self.add_loss(loss, inputs=inputs)
        return x_pred  # Just pass the prediction through

# Connect everything
vae_output = VAELossLayer()([input_img, z_decoded])

# Build model
vae = Model(input_img, vae_output)

# Compile and train
vae.compile(optimizer='adam')
vae.fit(x_train, x_train,
         epochs=10,
         batch_size=32,
         validation_split=0.2)

# =============================
# Visualize Results
# =============================

# Latent Space Mapping
mu, _, _ = encoder.predict(x_test)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(mu[:, 0], mu[:, 1], c=y_test, cmap='brg', s=2)
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.colorbar(scatter, label='Digit Class')
plt.title('Latent Space Representation of Digits')
plt.show()

# Single decoded image
sample_vector = np.array([[1.0, -1.0]])
decoded_example = decoder.predict(sample_vector)
decoded_image = decoded_example[0].reshape(img_width, img_height)
plt.figure()
plt.imshow(decoded_image, cmap='gray')
plt.title("Generated Image from Latent Vector [1.0, -1.0]")
plt.axis('off')
plt.show()

# Latent Grid Visualization
n = 20  # size of grid
figure = np.zeros((img_width * n, img_height * n, num_channels))

grid_x = np.linspace(-5, 5, n)
grid_y = np.linspace(-5, 5, n)[::-1]

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(img_width, img_height, num_channels)
        figure[i * img_width: (i + 1) * img_width,
               j * img_height: (j + 1) * img_height] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='gray')
plt.title("Latent Space Interpolation Grid")
plt.axis('off')
plt.show()