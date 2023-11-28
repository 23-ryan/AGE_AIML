from Models.base_ETH import base_ETH
import tensorflow as tf
import numpy as np
import os

class Bagging_ETH:
    
    def init_filename(self, model_path):
        for index in range(self.n_estimators):
            model = tf.keras.models.load_model(f"{model_path}/Bagging_ETH_{self.n_estimators}_{index}.keras")
            self.estimators.append(model)


    def __init__(self, image_input_shape = None, n_estimators=10, model_path = None):
        self.n_estimators = n_estimators
        self.estimators = []
        if(model_path != None):
            self.init_filename(model_path)
            return

        self.image_input_shape = image_input_shape

    def fit(self, train_images, train_eth, epochs=8):
        for _ in range(self.n_estimators):
            indices = np.random.choice(len(train_images), len(train_images), replace=True)
            X = train_images[indices]
            y = train_eth[indices]
            model = base_ETH(self.image_input_shape, dropout= 0.5, lr=.0001, batch_size=None)
            model.fit(X, y, epochs=epochs)
            self.estimators.append(model)

    def save(self, filename):
        for index, model in enumerate(self.estimators):
            model.model.save(f'{filename}_{self.n_estimators}_{index}.keras')

    def predict(self, test_images):
        result = [tree.predict(test_images) for tree in self.estimators]
        result = np.array(result)
        result  = np.argmax(result,axis=2)

        result = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=result)
        result = result.reshape((-1, 1))

        return result
    
    def evaluate(self, predictions, gt_values):
        predictions = predictions.reshape((-1,1))
        gt_values = gt_values.reshape((-1,1))

        acc = np.mean(predictions==gt_values)
        return acc
    
