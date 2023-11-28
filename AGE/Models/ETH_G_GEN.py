import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam


class ETH_G_GEN:
    def __init__(self, image_input_shape, lr=0.0001, batch_size=None, dropout=0.5):
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout

        gender_input_shape = (1,)
        image_input = Input(shape=image_input_shape, name='image_input')
        gender_input = Input(shape=gender_input_shape, name='gender_input')
        layers = Conv2D(32, (7, 7), activation='relu', input_shape=image_input_shape)(image_input)
        layers = BatchNormalization()(layers)
        layers = MaxPooling2D((2, 2))(layers)
        layers = Conv2D(64, (5, 5), activation='relu')(layers)
        layers = BatchNormalization()(layers)
        layers = MaxPooling2D((2, 2))(layers)
        layers = Conv2D(128, (5, 5), activation='relu')(layers)
        layers = Conv2D(128, (3, 3), activation='relu')(layers)
        layers = BatchNormalization()(layers)
        layers = MaxPooling2D((2, 2))(layers)
        flatten = Flatten()(layers)
        concatenated = Concatenate()([flatten, gender_input])
        dense1 = Dense(128, activation='relu')(concatenated)
        dense2 = Dropout(0.5)(dense1)
        output = Dense(5, activation='softmax')(dense2)


        self.model = Model(inputs=[image_input, gender_input], outputs=output)
        self.model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def fit(self, train_images, train_gender, train_eth, epochs=8):
        self.model.fit([train_images, train_gender], train_eth, epochs=epochs, batch_size=self.batch_size)

    def save(self, filename):
        self.model.save(f'{filename}_{self.batch_size}_{self.lr}_{self.dropout}.keras')
