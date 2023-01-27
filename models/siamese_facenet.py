import tensorflow as tf
from keras_facenet import FaceNet
from extract_features import load_embeddings_two_people, load_embeddings_same_person 

class SiameseFacenet(tf.keras.Model):
    def __init__(self):
        self.facenet = FaceNet()
        self.classifier = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def forward(self, x1, x2, method="euclidean"):
        x1 = self.facenet(x1)
        x2 = self.facenet(x2)
        result = None
        if method == "euclidean":
            # Compute distance
            distance = self.facenet.compute_distance(x1, x2)
            # Compute sigmoid
            result = tf.keras.activations.sigmoid(distance)
        else:
            result = self.classifier(x1, x2)
        return result

    def train(self, epochs=10, batch_size=32, learning_rate=0.001):
        """
        Train the classifier model.
        """
        # Load data
        X1, X2, y = load_embeddings_two_people()
