from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Flatten, Lambda
from keras.layers import Reshape
from keras.models import Model
from keras.losses import binary_crossentropy, mse
from keras.utils import plot_model
from keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

K.clear_session()

np.random.seed(237)


def process_data(data_path):

    data = np.load(data_path)

    # Min Max Scaling
    data = MinMaxScaler().fit_transform(data)

    X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)
    print('Shape train/test:', X_train.shape, X_test.shape)

    feature_length = X_train.shape[1]

    data = data.reshape((len(data), np.prod(data.shape[1:])))
    X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

    data = data.astype('float32')
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    return data, X_train, X_test, feature_length


def construct_vae(input_length, latent_dim, intermediate_dim):

    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def vae_loss(x, x_decoded_mean):
        xent_loss = binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        return xent_loss + kl_loss

    x = Input(shape=(input_length,))
    h = Dense(intermediate_dim, activation='relu')(x)

    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(input_length, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    decoder = Model(decoder_input, _x_decoded_mean)

    vae.compile(optimizer='rmsprop', loss=vae_loss)

    return vae, encoder, decoder


if __name__ == '__main__':

    is_train = False
    data_file = '../data/out/moment_features_5.npy'
    data, X_train, X_test, feature_len = process_data(data_file)

    latent_dim = 16
    intermediate_dim = 32
    batch_size = 32
    epochs = 50

    vae, encoder, decoder = construct_vae(feature_len, latent_dim, intermediate_dim)

    if is_train:
        history = vae.fit(X_train, X_train,
                          epochs=epochs,
                          shuffle=True,
                          batch_size=batch_size,
                          validation_data=(X_test, X_test),
                          verbose=2)
        vae.save_weights('vae_dnn.h5')

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('vae_dnn_train.jpeg')
        plt.show()

    else:
        vae.load_weights('vae_dnn.h5')

    # Transform to latent representation
    encoded_data = encoder.predict(data, batch_size=batch_size)

    pd.DataFrame(encoded_data).to_csv('latest_rep_dnn.csv', index=None)

    print('Completed.')
