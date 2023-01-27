from models.siamese_classifier import SiameseClassifier
import tensorflow as tf
from dataset.data_loader import CustomDataLoader

def main():
    # Load the model
    model = SiameseClassifier()
    model = model.model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # Data loader
    batch_size = 64
    data_loader = CustomDataLoader(batch_size=batch_size)
    # Train
    model.fit(data_loader, epochs=10, batch_size=batch_size)

if __name__ == "__main__":
    main()
