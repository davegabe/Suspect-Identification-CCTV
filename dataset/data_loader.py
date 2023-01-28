import tensorflow as tf
from dataset.extract_features import load_embeddings

class CustomDataLoader(tf.keras.utils.Sequence):
    def __init__(self, batch_size, input_size=1024, is_train=True, train_partition=1):
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = False
        self.is_train = is_train
        self.train_partition = train_partition
        self.n = 10000

    def __getitem__(self, index):
        X, y = load_embeddings(self.batch_size, self.is_train, self.train_partition)
        return X, y

    def __len__(self):
        return self.n // self.batch_size
