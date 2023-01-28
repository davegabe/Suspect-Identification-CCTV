from keras_facenet import FaceNet
import os
import numpy as np

pathLFW = './LFW/lfw_funneled/'
pathEmbeddings = '../embeddings/'


def extract_features():
    """
    Extracts the embeddings for every image in the LFW dataset and saves them in a numpy array for each person.
    """
    # Load the FaceNet model
    facenet = FaceNet()
    os.makedirs(pathEmbeddings, exist_ok=True)
    # For every image in pathLFW
    for person in os.listdir(pathLFW):
        # Skip if not a directory
        if not os.path.isdir(os.path.join(pathLFW, person)):
            continue
        # Embeddings for the person
        embeddings = []
        # For every image in the person folder
        images = os.listdir(os.path.join(pathLFW, person))
        for image in images:
            embedding = facenet.extract(os.path.join(pathLFW, person, image))
            embeddings.append(embedding)
        # Save the embeddings
        np.save(os.path.join(pathEmbeddings, person + '.npy'), embeddings)

if __name__ == '__main__':
    extract_features()