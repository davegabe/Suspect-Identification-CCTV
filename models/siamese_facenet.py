import tensorflow as tf
from keras_facenet import FaceNet
import numpy as np
import os

class SiameseFacenet(object):
    def __init__(self):
        self.facenet1 = FaceNet()
        self.facenet2 = FaceNet()
        self.classifier = tf.keras.models.Sequential([]) # TODO: Add classifier
        
if __name__ == "__main__":
    siamese_facenet = SiameseFacenet()