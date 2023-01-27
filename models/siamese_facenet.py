import tensorflow as tf
from keras_facenet import FaceNet
from extract_features import load_embeddings
from tqdm import tqdm

# class SiameseFacenet(tf.keras.Model):
#     def __init__(self, input_shape=(32, 512)):
#         super().__init__()
#         self.in_shape = input_shape
#         self.facenet = FaceNet()
#         self.siamese_classifier = SiameseClassifier(input_shape=(32, 512))

#     def model(self):
#         x = tf.keras.Input(shape=self.in_shape)
#         return tf.keras.Model(inputs=x, outputs=self.call(x))

#     def call(self, x, method="euclidean"):
#         # x1 = self.facenet.extract(x1)
#         # x2 = self.facenet.extract(x2)
#         x1 = x[0]
#         x2 = x[1]
#         result = 1
#         # if method == "euclidean":
#         #     # Compute distance
#         #     # distance = list(map(lambda x: self.facenet.compute_distance(x[0], x[1]), zip(x1, x2))) #TODO: Check if this is correct
#         #     distance = tf.keras.losses.MSE(x1, x2)
#         #     # Compute sigmoid
#         #     result = tf.keras.activations.sigmoid(distance)
#         # else:
#         #     result = self.classifier(x1, x2)
#         return result

#     def fit(self, epochs=10, batch_size=32, learning_rate=0.001):
#         """
#         Train the classifier model.
#         """
#         # For each epoch
#         for epoch in tqdm(range(epochs)):
#             # Load batch
#             x1, x2, y = load_embeddings(batch_size)
#             # Train
#             with tf.GradientTape() as tape:
#                 # Forward
#                 y_pred = self.call(x1, x2)
#                 # Compute loss
#                 loss = tf.keras.losses.binary_crossentropy(y, y_pred)
#                 # Compute gradients
#                 gradients = tape.gradient(loss, self.classifier.trainable_variables)
#                 # Update weights
#                 optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#                 optimizer.apply_gradients(zip(gradients, self.classifier.trainable_variables))
#             # Print loss
#             print("Epoch: {}, Loss: {}".format(epoch, loss))
