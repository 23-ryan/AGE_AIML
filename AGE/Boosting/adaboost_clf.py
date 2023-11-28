import numpy as np
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier

class AdaBoostClassifierTF:
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

    def fit(self, X, y, n_estimators, k=2):
        """
        Fit the AdaBoost model with n_estimators iterations.

        Parameters:
        - X: 2D array, shape (n_samples, n_features), Input features.
        - y: 1D array, shape (n_samples,), Response labels (±1).
        - n_estimators: Number of boosting iterations.

        Returns:
        - The fitted AdaBoostClassifier.
        """

        n_samples, n_features = X.shape

        # Initialize TensorFlow variables
        self.sample_weights = tf.Variable(tf.ones((n_samples, 1), dtype=tf.float64) / n_samples, trainable=False)
        self.weak_learners = [DecisionTreeClassifier(max_depth=3, random_state=i) for i in range(n_estimators)]
        self.learner_weights = tf.Variable(tf.zeros((n_estimators, 1), dtype=tf.float64), trainable=False)
        self.errors = tf.Variable(tf.zeros((n_estimators, 1), dtype=tf.float64), trainable=False)

        # For each iteration
        for t in range(n_estimators):
            # Create a weak learner (stump) and fit it with weighted samples
            self.weak_learners[t].fit(X, y, sample_weight=np.ravel(self.sample_weights.numpy()))

            # Make predictions with the weak learner
            y_pred = self.weak_learners[t].predict(X)

            # Calculate weighted error and learner weight
            error = tf.reduce_sum(self.sample_weights * tf.cast((y_pred != y), dtype=tf.float64))
            error = error/tf.reduce_sum(self.sample_weights)
            self.errors[t] = error

            self.learner_weights[t] = (tf.math.log((1 - error) / error) / 2) + tf.math.log(k-1)

            # Update sample weights based on the weighted error
            exp_term = tf.exp(-self.learner_weights[t] * y * y_pred)
            self.sample_weights.assign((self.sample_weights * exp_term) / tf.reduce_sum(self.sample_weights * exp_term))

        return self

    def predict(self, X):
        """
        Make predictions using the already fitted AdaBoost model.

        Parameters:
        - X: 2D array, shape (n_samples, n_features), Input features for predictions.

        Returns:
        - Predicted class labels, 1D array of shape (n_samples,).
        """
        y_pred = tf.zeros(X.shape[0], dtype=tf.float64)
        for t in range(len(self.weak_learners)):
            y_pred += self.learner_weights[t] * tf.cast(self.weak_learners[t].predict(X), dtype=tf.float64)
        y_pred = tf.sign(y_pred)
        return tf.squeeze(y_pred)

