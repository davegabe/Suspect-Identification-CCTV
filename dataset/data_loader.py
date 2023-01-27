import tensorflow as tf
from extract_features import load_embeddings

class CustomDataLoader(tf.keras.utils.Sequence):
    def __init__(self, batch_size, input_size=512):
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = False
        self.n = 1000

    def __getitem__(self, index):
        X, y = load_embeddings(self.batch_size)
        return X, y

    def __len__(self):
        return self.n // self.batch_size
