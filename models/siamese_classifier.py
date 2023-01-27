import tensorflow as tf

class SiameseClassifier(tf.keras.Model):
    def __init__(self, input_shape=1024):
        super().__init__()
        self.in_shape = input_shape
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

    def model(self):
        x = tf.keras.Input(shape=self.in_shape)
        return tf.keras.Model(inputs=x, outputs=self.call(x))

    def call(self, x: tf.Tensor):
        y = self.classifier(x)
        return y