from models.siamese_classifier import SiameseClassifier
from dataset.data_loader import CustomDataLoader
import tensorflow as tf
import numpy as np
import os

model_path = "models/siamese_classifier.h5"

def hardcode_test(model, test_data_loader):
    for i in range(10):
        print("Different people")
        person_name1 = "Zhu_Rongji.npy"   #LFW/lfw_funneled/Zhu_Rongji
        embeddings = test_data_loader.get_by_name(person_name1)
        embedding1 = embeddings[np.random.randint(0, len(embeddings))]
        person_name2 = "Yoshiyuki_Kamei.npy"
        embeddings = test_data_loader.get_by_name(person_name2)
        embedding2 = embeddings[np.random.randint(0, len(embeddings))]
        dist = test_data_loader.euclidean_distance(embedding1, embedding2)
        ypred = model.predict(dist)
        print("ypred: ", ypred)

        print("Same people")
        person_name1 = "Zhu_Rongji.npy"
        embeddings = test_data_loader.get_by_name(person_name2)
        embedding1 = embeddings[np.random.randint(0, len(embeddings))]
        embedding2 = embeddings[np.random.randint(0, len(embeddings))]
        dist = test_data_loader.euclidean_distance(embeddings[0], embeddings[1])
        ypred = model.predict(dist)
        print("ypred: ", ypred)

        print("\n\n")


def main():
    # Load the model
    model = SiameseClassifier()
    model = model.model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    # Data loader
    batch_size = 256
    epochs = 2
    train_partition = 0.8
    train_data_loader = CustomDataLoader(batch_size=batch_size, is_train=True, train_partition=train_partition)
    test_data_loader = CustomDataLoader(batch_size=batch_size, is_train=False, train_partition=train_partition)
    # Load the model or train it if it doesn't exist
    # if os.path.exists(model_path):
    #     # Load the modelc
    #     model.load_weights(model_path)
    # else:
        # Train
    model.fit(train_data_loader, epochs=epochs, batch_size=batch_size, validation_data=test_data_loader)
    # Save the model
    model.save(model_path)
    # Hard-coded test
    #hardcode_test(model, test_data_loader)  
    

if __name__ == "__main__":
    main()
