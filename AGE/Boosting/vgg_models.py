import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import os
import numpy as np
from tqdm import tqdm
import cv2

from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout

class VGG_GEN:

    def __init__(self, lr=0.0001, backbone_output=256):
        base_model = VGG16()
        base_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

        print(base_model.summary())
        for layer in base_model.layers:
            layer.trainable = False
        self.model = Sequential()
        self.model.add(base_model)
        self.model.add(layers.Dense(2048, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        # model.add(layers.Dense(512, activation='relu'))
        # model.add(layers.Dropout(0.3))
        self.model.add(layers.Dense(backbone_output, activation='relu', name = "face_features"))
        self.model.add(layers.Dense(1, activation='sigmoid'))

        print(self.model.summary())

        self.model.compile(optimizer=Adam(learning_rate=.0001), loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self, train_images, train_gender, epochs=10, batch_size=128):
        self.model.fit(train_images, train_gender, epochs=epochs, batch_size=batch_size)
    
    def save(self, filepath):
        self.model.save(filepath)

    def evaluate(self, X, y):
        result = self.model.predict(X)
        print("result", result.shape)
        y = y.reshape((-1,1))
        result[result<=0.5] = 0
        result[result > 0.5] = 1
        val = np.mean(result==y, axis=0)
        return val

    

class VGG_ETH:
    def __init__(self, lr=0.0001, backbone_output=256) -> None:

        base_model = VGG16()
        base_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

        print(base_model.summary())
        for layer in base_model.layers:
            layer.trainable = False
        self.model = Sequential()
        self.model.add(base_model)
        self.model.add(layers.Dense(2048, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        # model.add(layers.Dense(512, activation='relu'))
        # model.add(layers.Dropout(0.3))
        self.model.add(layers.Dense(backbone_output, activation='relu', name = "face_features"))
        self.model.add(layers.Dense(5))

        print(self.model.summary())

        self.model.compile(optimizer=Adam(learning_rate=.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def init_base_model(self):
        for layer in self.base_model.layers:
            layer.trainable = False

    def fit(self, train_images, train_gender, epochs=10, batch_size=128):
        self.model.fit(train_images, train_gender, epochs=epochs, batch_size=batch_size)
    
    def save(self, filepath):
        self.model.save(filepath)

    def evaluate(self, X, y):
        result = self.model.predict(X)
        ans = np.argmax(result, axis=1)
        # y = y.reshape((-1,1))
        val = np.mean(y==ans)
        return val