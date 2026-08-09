import numpy as np
import tensorflow as tf

class KNNClusterAssigner:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, cluster_labels):
        self.X = tf.constant(X, dtype=tf.float32)
        self.labels = tf.constant(cluster_labels, dtype=tf.int32)
        self.n_clusters = int(cluster_labels.max()) + 1
        return self

    def predict(self, X_new):
        X_new = tf.constant(X_new, dtype=tf.float32)
        diff = X_new[:, None, :] - self.X[None, :, :]
        dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=2))

        _, idx = tf.math.top_k(-dist, k=self.k)
        neighbor_labels = tf.gather(self.labels, idx)

        onehot = tf.one_hot(neighbor_labels, depth=self.n_clusters)
        votes = tf.reduce_sum(onehot, axis=1)

        return tf.argmax(votes, axis=1).numpy().astype("int32")