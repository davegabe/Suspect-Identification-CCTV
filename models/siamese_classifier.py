import tensorflow as tf
from tqdm import tqdm

from extract_features import load_embeddings

class SiameseClassifier(tf.keras.Model):
    def __init__(self, input_shape=(2, 512)):
        super().__init__()
        self.in_shape = input_shape
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

    def model(self):
        x = tf.keras.Input(shape=self.in_shape)
        return tf.keras.Model(inputs=x, outputs=self.call(x))

    def call(self, x: tf.Tensor):
        y = self.classifier(x)
        return y