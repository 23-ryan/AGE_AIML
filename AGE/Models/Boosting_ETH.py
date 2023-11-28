import numpy as np
import pickle
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Model

class Boosting_ETH:
    def __init__(self):
        """
        Initialize an AdaBoost Classifier.

        Attributes:
        - weak_learners: List of weak learner models.
        - learner_weights: Weights assigned to weak learners (alpha_t's).
        - errors: Error rates of each iteration. (epsilon_t's)
        - sample_weights: Weight distribution over training samples. (w_t(i))
        """
        self.weak_learners = []
        self.learner_weights = []
        self.errors = []
        self.sample_weights = None

    def fit(self, n_estimators, backbone_output=128, k=5):
        """
        Fit the AdaBoost model with n_estimators iterations.

        Parameters:
        - X: 2D array, shape (n_samples, n_features), Input features.
        - y: 1D array, shape (n_samples,), Response labels (±1).
        - n_estimators: Number of boosting iterations.

        Returns:
        - The fitted AdaBoostClassifier.
        """
        features_path_X = f'Saved_Models/Features/vgg_transfer_eth_{backbone_output}_X.keras.npy'
        X = np.load(features_path_X)

        features_path_Y = f'Saved_Models/Features/vgg_transfer_eth_{backbone_output}_Y.keras.npy'
        feature_Y = np.load(features_path_Y)
        y= feature_Y
        # y = np.where(feature_Y == 0, -1, 1)

        n_samples, n_features = X.shape
        print("Num samples", n_samples)

        # Initialize TensorFlow variables
        self.sample_weights = tf.Variable(tf.ones((n_samples, 1), dtype=tf.float64) / n_samples, trainable=False)
        self.weak_learners = [DecisionTreeClassifier(max_depth=3, random_state=i) for i in range(n_estimators)]
        self.learner_weights = tf.Variable(tf.zeros((n_estimators, 1), dtype=tf.float64), trainable=False)
        self.errors = tf.Variable(tf.zeros((n_estimators, 1), dtype=tf.float64), trainable=False)

        # For each iteration
        for t in range(n_estimators):
            # Create a weak learner (stump) and fit it with weighted samples
            # print(self.sample_weights)
            self.weak_learners[t].fit(X, y, sample_weight=np.ravel(self.sample_weights.numpy()))

            # Make predictions with the weak learner
            y_pred = self.weak_learners[t].predict(X)

            # Calculate weighted error and learner weight
            sign = tf.cast((y_pred != y), dtype=tf.float64)
            # print(sign.shape)
            sign = tf.reshape(sign, [sign.shape[0], 1])
            # print(self.sample_weights * sign)
            error = tf.reduce_sum(self.sample_weights * sign)
            error = error/ tf.reduce_sum(self.sample_weights)
            self.errors[t].assign(error)
            # print(error)
            k = tf.cast(k, dtype=tf.float64)

            self.learner_weights[t].assign((tf.math.log((1 - error) / error) / 2)+ tf.math.log(k-1))

            # Update sample weights based on the weighted error
            # print(self.learner_weights[t])
            # max_exp = tf.reduce_max(-self.learner_weights[t] * y * y_pred)
            # print("max expo",max_exp)
            exp_term = tf.exp((self.learner_weights[t] * sign))
            # print(exp_term)
            shape1= exp_term.shape[0]
            exp_term  = tf.reshape(exp_term, [shape1, 1])
            self.sample_weights.assign((self.sample_weights * exp_term) / tf.reduce_sum(self.sample_weights * exp_term))

        return self

    def predict(self, X, k=5):
        """
        Make predictions using the already fitted AdaBoost model.

        Parameters:
        - X: 2D array, shape (n_samples, n_features), Input features for predictions.

        Returns:
        - Predicted class labels, 1D array of shape (n_samples,).
        """
        y_pred = [[tf.zeros(X.shape[0], dtype=tf.float64)] for x in range(k)]
        for t in range(len(self.weak_learners)):
            for itr in range(k):
                y_pred[itr] =  y_pred[itr] + self.learner_weights[t] * tf.cast(self.weak_learners[t].predict(X)==itr, dtype=tf.float64)
        
        y_pred = tf.stack(y_pred)
        y_pred = tf.argmax(y_pred, axis=0)
        # y_pred = tf.sign(y_pred)

        return tf.squeeze(y_pred)
    
    def evaluate(self, X, y, backbone_output=128):
        # y_pred = tf.zeros(X.shape[0], dtype=tf.float64)
        # for t in range(len(self.weak_learners)):
        #     y_pred += self.learner_weights[t] * tf.cast(self.weak_learners[t].predict(X), dtype=tf.float64)
        # y_pred = tf.sign(y_pred)
        # return tf.squeeze(y_pred)

        model_path = f'Saved_Models/vgg_transfer_eth_{backbone_output}.keras'
        vgg_eth_model = tf.keras.models.load_model(model_path)
        feature_layer_model = Model(inputs=vgg_eth_model.input, outputs=vgg_eth_model.get_layer("face_features").output)
        features = feature_layer_model.predict(X)

        predict = self.predict(features)
        # y = np.where(y == 0, -1, 1)
        acc = tf.math.reduce_mean(tf.cast(predict == y, tf.float32))
        return acc
    
    def load(self, filename, n_estimators, backbone_output):
        file_name = f'{filename}_{n_estimators}_{backbone_output}.pickle'
        with open(file_name, 'rb') as file:
            loaded_object = pickle.load(file)

        self.weak_learners = loaded_object['weak_learner']
        self.learner_weights = loaded_object['learner_weights']


    
    def save(self, filename, n_estimators, backbone_output):
        Obj = {'weak_learner': self.weak_learners, 'learner_weights': self.learner_weights}
        file_name = f'{filename}_{n_estimators}_{backbone_output}.pickle'
        # print("done")
        with open(file_name, 'wb') as file:
            pickle.dump(Obj, file)
    


