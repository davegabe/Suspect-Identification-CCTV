from models.siamese_classifier import SiameseClassifier
from dataset.data_loader import CustomDataLoader

def main():
    # Load the model
    model = SiameseClassifier()
    model = model.model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # Data loader
    batch_size = 32
    epochs = 50
    train_data_loader = CustomDataLoader(batch_size=batch_size, is_train=True, train_partition=0.8)
    test_data_loader = CustomDataLoader(batch_size=batch_size, is_train=False, train_partition=0.8)
    # Train
    model.fit(train_data_loader, epochs=epochs, batch_size=batch_size, validation_data=test_data_loader)

if __name__ == "__main__":
    main()
