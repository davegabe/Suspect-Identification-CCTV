from extract_features import load_embeddings
# from models.siamese_facenet import SiameseFacenet
from models.siamese_classifier import SiameseClassifier
import tensorflow as tf

def main():
    # Load the model
    model = SiameseClassifier()
    model = model.model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # Load the data
    x, y = load_embeddings(32)
    # Train
    model.fit(x=x, y=y, epochs=10, batch_size=32)

if __name__ == "__main__":
    main()
