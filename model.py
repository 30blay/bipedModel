from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Input, Lambda, Dense, Dropout
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
import os

weights_path = os.path.dirname(__file__) + '/model_weights.h5'
input_shape = (224, 224, 3)


class BipedModel:
    def __init__(self, load_weights=True):
        self.encoder = self.init_encoder()
        self.latent_shape = (self.encoder.output_shape[1],)

        self.siamese = self.init_siamese(load_weights)

    def init_encoder(self):
        input_tensor = Input(shape=input_shape)
        model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_tensor=input_tensor,
            input_shape=input_shape,
            pooling='avg')
        return model

    def init_siamese(self, load_weights=True):
        left_input = Input(self.latent_shape)
        right_input = Input(self.latent_shape)

        # merge two encoded inputs with the l1 distance between them
        L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]), output_shape=self.latent_shape)
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

    def get_encoder(self):
        return self.encoder

    def get_siamese(self):
        return self.siamese

    def extract_features(self, img_path):
        from keras.preprocessing import image
        from keras.preprocessing.image import ImageDataGenerator
        import numpy as np

        datagen = ImageDataGenerator(rescale=1. / 255)

        img = image.load_img(img_path, target_size=input_shape[0:2])
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        generator = datagen.flow(x)

        for inputs_batch in generator:
            features_batch = self.encoder.predict(inputs_batch)
            return features_batch

    def get_similarity(self, feature1, feature2):
        similarity = self.siamese.predict([feature1, feature2])
        return similarity[0][0]
