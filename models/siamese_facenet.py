import tensorflow as tf
from keras_facenet import FaceNet
from models.siamese_classifier import SiameseClassifier
import numpy as np

facenet = FaceNet()

class SiameseFacenet():
    def __init__(self):
        super().__init__()
        self.siamese_classifier = SiameseClassifier()

    def forward(self, x1, x2):
        x1 = facenet.extract(x1)
        x2 = facenet.extract(x2)
        # result = self.classifier.predict(euclidean_distance(x1, x2))
        return None
