import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU, BatchNormalization
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
from keras.models import load_model
import cv2

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Parameters
img_rows = 224
img_cols = 224
channels = 3
img_shape = (img_rows, img_cols, channels)
latent_dim = 100

# Generator
def build_generator():
    model = Sequential()
    
    # Starting with a dense layer
    model.add(Dense(7 * 7 * 256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Reshape((7, 7, 256)))
    
    # Upsampling to 14x14
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    
    # Upsampling to 28x28
    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    
    # Upsampling to 56x56
    model.add(Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    
    # Upsampling to 112x112
    model.add(Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    
    # Final upsampling to 224x224
    model.add(Conv2DTranspose(8, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    
    # Output layer
    model.add(Conv2DTranspose(channels, (3, 3), activation='tanh', padding='same'))
    
    return model

# Discriminator
def build_discriminator():
    model = Sequential()
    
    # Input layer
    model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', input_shape=img_shape))
    model.add(LeakyReLU(alpha=0.2))
    
    # Downsampling
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    
    # Downsampling
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    
    # Downsampling
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    
    # Downsampling
    model.add(Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    
    # Flatten and output
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    return model

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(0.0002, 0.5),
                      metrics=['accuracy'])

# Build the generator
generator = build_generator()

# Combined GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return model

gan = build_gan(generator, discriminator)

def load_real_samples():
    X = np.load('model/malignant_X.npy')
    X = X.astype('float32')
    X = (X - 0.5) / 0.5  # Normalize to [-1, 1]
    return X

# Training function
def train(epochs, X_train, batch_size, sample_interval):
    # Load your dataset here
    # For demonstration, we'll use random data
    # Replace this with your actual data loading code
    
    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    for epoch in range(epochs):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        # Select a random batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        
        # Generate a batch of new images
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)
        
        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        
        # Train the generator (to have the discriminator label samples as valid)
        g_loss = gan.train_on_batch(noise, valid)
        
        # Print progress
        print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")
        
def getVanillaCoGan():
    if os.path.exists('model/benign_model.keras'):
        malign_gan_model = load_model('model/malignant_model.keras')
        benign_gan_model = load_model('model/benign_model.keras')
    else:
        # load image data
        dataset = load_real_samples()
        # train model
        train(100, dataset, 32, 200)
        #generator.save('/content/drive/MyDrive/model/malignant_model.keras')
    return malign_gan_model, benign_gan_model

def generateImages(model, img, img_type, index):
    r, c = 5, 5
    img = img.astype('float32')
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = model.predict(noise)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gen_imgs = 0.5 * gen_imgs + 0.5
    cnt = 0
    for i in range(r):
        for j in range(c):
            generates = gen_imgs[cnt]
            generates = cv2.cvtColor(generates, cv2.COLOR_BGR2RGB)
            generates = generates.astype('float32')
            new_image = cv2.addWeighted(img, 0.5, generates, 0.5, 0)
            cv2.imwrite("GanDataset/"+img_type+"/"+str(index)+"_"+str(cnt)+".jpg", new_image)
            cnt += 1


