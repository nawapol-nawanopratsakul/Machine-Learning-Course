import numpy as np
import tensorflow as tf

class TFKNNClassifier:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = tf.constant(X, dtype=tf.float32)
        self.y_train = tf.constant(y, dtype=tf.int32)
        self.n_classes = int(y.max()) + 1
        return self

    def _distance(self, X_new):
        diff = X_new[:, None, :] - self.X_train[None, :, :]
        return tf.sqrt(tf.reduce_sum(tf.square(diff), axis=2))

    def predict(self, X):
        X = tf.constant(X, dtype=tf.float32)
        dist = self._distance(X)
        _, idx = tf.math.top_k(-dist, k=self.k)
        neighbor_labels = tf.gather(self.y_train, idx)
        onehot = tf.one_hot(neighbor_labels, depth=self.n_classes)
        votes = tf.reduce_sum(onehot, axis=1)
        return tf.argmax(votes, axis=1).numpy()

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))