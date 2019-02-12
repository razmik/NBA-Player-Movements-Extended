from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

K.clear_session()

np.random.seed(237)


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def process_data(data_path):

    data = np.load(data_path)

    X_train, X_test = train_test_split(data, test_size=0.05, random_state=42)
    print('Shape train/test:', X_train.shape, X_test.shape)

    image_size = X_train.shape[1], X_train.shape[2]

    data = np.reshape(data, [-1, image_size[0], image_size[1], 1])
    X_train = np.reshape(X_train, [-1, image_size[0], image_size[1], 1])
    X_test = np.reshape(X_test, [-1, image_size[0], image_size[1], 1])

    data = data.astype('float32') / 255
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    return data, X_train, X_test, image_size


def construct_vae(image_size, kernel_size, latent_dim):
    # network parameters
    input_shape = (image_size[0], image_size[1], 1)

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    x = Conv2D(filters=16, kernel_size=kernel_size, activation='relu', strides=1, padding='same')(x)
    x = Conv2D(filters=32, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)
    x = Conv2D(filters=64, kernel_size=kernel_size, activation='relu', strides=1, padding='same')(x)

    # shape info needed to build decoder model
    shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    x = Conv2DTranspose(filters=64, kernel_size=kernel_size, activation='relu', strides=1, padding='same')(x)
    x = Conv2DTranspose(filters=32, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(filters=16, kernel_size=kernel_size, activation='relu', strides=1, padding='same')(x)

    outputs = Conv2DTranspose(filters=1,
                              kernel_size=kernel_size,
                              activation='sigmoid',
                              padding='same',
                              name='decoder_output')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    # VAE loss = mse_loss or xent_loss + kl_loss
    reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))

    reconstruction_loss *= image_size[0] * image_size[1]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()
    plot_model(vae, to_file='vae_cnn.png', show_shapes=True)

    return vae, encoder, decoder


if __name__ == '__main__':

    is_train = False
    data_file = '../data/out/moment_frames_5.npy'
    data, X_train, X_test, im_size = process_data(data_file)

    kernel_size = (3, 3)
    latent_dim = 128
    batch_size = 128
    epochs = 10

    vae, encoder, decoder = construct_vae(im_size, kernel_size, latent_dim)

    if is_train:
        history = vae.fit(X_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_data=(X_test, None),
                          verbose=2)
        vae.save_weights('vae_cnn.h5')

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('vae_train.jpeg')
        plt.show()

    else:
        vae.load_weights('vae_cnn.h5')

    # Transform to latent representation
    encoded_data = encoder.predict(data, batch_size=batch_size)

    pd.DataFrame(encoded_data[0]).to_csv('latest_rep_cnn.csv', index=None)

    print('Completed.')
