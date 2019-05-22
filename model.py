from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Input, Lambda, Dense, Dropout
from keras.models import Model
from keras import backend as K
from keras import models
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
import os

weights_path = os.path.dirname(__file__) + '/model_weights.h5'


def get_encoder():
    input_tensor = Input(shape=(224, 224, 3))
    model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(224, 224, 3),
        pooling='avg')
    return model


latent_shape = (get_encoder().output_shape[1],)


def get_siamese(load_weights=True):
    left_input = Input(latent_shape)
    right_input = Input(latent_shape)

    # merge two encoded inputs with the l1 distance between them
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]), output_shape=latent_shape)
    L1_distance = L1_layer([left_input, right_input])
    normalised_distance = BatchNormalization()(L1_distance)
    prediction = Dense(1, activation='sigmoid')(normalised_distance)
    prediction = Dropout(0.4)(prediction)

    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    optimizer = Adam(lr=0.0001)
    siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer)

    if load_weights:
        siamese_net.load_weights(weights_path)

    return siamese_net
