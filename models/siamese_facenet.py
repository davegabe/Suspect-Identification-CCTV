import tensorflow as tf
from keras_facenet import FaceNet
from models.siamese_classifier import SiameseClassifier

class SiameseFacenet():
    def __init__(self):
        super().__init__()
        self.facenet = FaceNet()
        self.siamese_classifier = SiameseClassifier()

    def forward(self, x1, x2, method="euclidean"):
        x1 = self.facenet.extract(x1)
        x2 = self.facenet.extract(x2)
        result = None
        if method == "euclidean":
            # Compute distance
            distance = list(map(lambda x: self.facenet.compute_distance(x[0], x[1]), zip(x1, x2))) #TODO: Check if this is correct
            # Compute sigmoid
            result = tf.keras.activations.sigmoid(distance)
        else:
            result = self.classifier(x1, x2)
        return result
