from models.siamese_facenet import SiameseFacenet
def main():
    # Load the model
    model = SiameseFacenet()
    # Train the model
    epochs = 100
    model.train(epochs)


if __name__ == "__main__":
    main