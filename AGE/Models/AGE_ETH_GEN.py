import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import Adam


class AGE_ETH_GEN:
    def __init__(self, image_input_shape, lr=0.0001, batch_size=None, dropout=0.5):
        self.image_input_shape = image_input_shape
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        # gender_input_shape = (1,)
        image_input = Input(shape=self.image_input_shape, name='image_input')
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

        gender_dense1 = Dense(128, activation='relu')(flatten)
        gender_dense2 = Dropout(dropout)(gender_dense1)
        output_gender = Dense(1, activation='sigmoid', name='gender_output')(gender_dense2)

        eth_concatenated = Concatenate()([flatten, output_gender])
        eth_dense1 = Dense(128, activation='relu')(eth_concatenated)
        eth_dense2 = Dropout(dropout)(eth_dense1)
        output_eth = Dense(5, activation='softmax', name='eth_output')(eth_dense2)

        age_con = Concatenate()([flatten, output_gender])
        age_dense1 = Dense(128, activation='relu')(age_con)
        age_dense2 = Dropout(dropout)(age_dense1)
        output_age = Dense(1, activation='relu', name='age_output')(age_dense2)
        self.model = Model(inputs=image_input, outputs=[output_gender, output_age, output_eth])
        self.model.compile(optimizer=Adam(learning_rate=lr),
            loss={'gender_output': 'binary_crossentropy', 'age_output': 'mae', 'eth_output': 'sparse_categorical_crossentropy'},
            loss_weights={'gender_output': 2, 'age_output': 5, 'eth_output': 4},
            metrics={'gender_output': 'accuracy', 'age_output': 'mae', 'eth_output': 'accuracy'}
        )

    def fit(self, train_images, train_gender, train_age, train_eth, epochs = 8):
        train_label = {"gender_output" : train_gender, "age_output" : train_age, "eth_output" : train_eth}
        self.model.fit(train_images, train_label, epochs=epochs, batch_size=self.batch_size, verbose=1)

    def save(self, filename):
        self.model.save(f'{filename}_{self.batch_size}_{self.lr}_{self.dropout}.keras')