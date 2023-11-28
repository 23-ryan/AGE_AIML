import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam


class base_GEN:
    def __init__(self, image_input_shape, dropout= 0.5, lr=.0001, batch_size=32):
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.model = Sequential([
            Conv2D(32, (7, 7), activation='relu', input_shape=image_input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(64, (5, 5), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(128, (5, 5), activation='relu'),
            Conv2D(128, (3, 3), activation='relu'),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(dropout),
            Dense(1, activation='sigmoid'),
        ])
        self.model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self, train_images, train_gender, epochs=8):
        self.model.fit(train_images, train_gender, epochs=epochs, batch_size=self.batch_size)

    def save(self, filename):
        self.model.save(f'{filename}_{self.batch_size}_{self.lr}_{self.dropout}.keras')
