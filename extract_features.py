from keras_facenet import FaceNet
import os
import numpy as np

pathLFW = './LFW/lfw_funneled/'
pathEmbeddings = './embeddings/'


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


def load_embeddings_two_people():
    """
    Loads two people embeddings and returns them.
    """
    # Get all people
    people = filter(lambda x: x.endswith('.npy'), os.listdir(pathEmbeddings))
    person1 = np.random.choice(people)
    while True:
        person2 = np.random.choice(people)
        if person1 != person2:
            break
    # Load the embeddings
    embeddings1 = np.load(os.path.join(pathEmbeddings, person1))
    embeddings2 = np.load(os.path.join(pathEmbeddings, person2))
    # Return random embeddings from the two people
    return np.random.choice(embeddings1), np.random.choice(embeddings2), 0

def load_embeddings_same_person():
    """
    Loads two embeddings from the same person and returns them.
    """
    # Get all people
    people = filter(lambda x: x.endswith('.npy'), os.listdir(pathEmbeddings))
    # Pick a random person having more than one embedding
    while True:
        person = np.random.choice(people)
        embeddings = np.load(os.path.join(pathEmbeddings, person))
        if len(embeddings) > 1:
            break
    # Return two random embeddings from the same person
    return np.random.choice(embeddings), np.random.choice(embeddings), 1


if __name__ == '__main__':
    extract_features()