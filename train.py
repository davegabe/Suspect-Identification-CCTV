from models.siamese_classifier import SiameseClassifier
from dataset.data_loader import CustomDataLoader
from dataset.extract_features import random_embedding_from_npy
import numpy as np
import os

model_path = "models/siamese_classifier.h5"

def hardcode_test(model):
    for i in range(10):

        print("Different people")
        person_name1 = "Zhu_Rongji"   #LFW/lfw_funneled/Zhu_Rongji
        embedding1 = random_embedding_from_npy(os.path.join("./embeddings", person_name1+".npy"))
        person_name2 = "Yoshiyuki_Kamei"
        embedding2 = random_embedding_from_npy(os.path.join("./embeddings", person_name2+".npy"))
        concat = np.array([np.concatenate((embedding1, embedding2))])
        ypred = model.predict(concat)
        print("ypred: ", ypred)

        print("Same people")
        person_name1 = "Zhu_Rongji"
        embedding1 = random_embedding_from_npy(os.path.join("./embeddings", person_name1+".npy"))
        person_name2 = "Zhu_Rongji"
        embedding2 = random_embedding_from_npy(os.path.join("./embeddings", person_name2+".npy"))
        
        concat = np.array([np.concatenate((embedding1, embedding2))])
        ypred = model.predict(concat)
        print("ypred: ", ypred)

        print("\n\n\n\n")


def main():
    # Load the model
    model = SiameseClassifier()
    model = model.model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # Data loader
    batch_size = 32
    epochs = 50
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
    hardcode_test(model)    
    

if __name__ == "__main__":
    main()
