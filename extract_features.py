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

def load_npy(path: str) -> np.ndarray:
    """
    Loads a numpy array from a path.
    """
    # Load the numpy array
    array = np.load(path, allow_pickle=True)[0]
    # Get random embedding
    data: dict = array[np.random.randint(0, len(array))]
    # Return embedding
    embedding = data["embedding"]
    return embedding

def load_embeddings(batch_size=32):
    """
    Loads two people embeddings and returns them.
    """
    # Get all people
    people = filter(lambda x: x.endswith('.npy'), os.listdir(pathEmbeddings))
    people = np.array(list(people))
    # Get half the batch size
    half_batch_size = batch_size // 2
    # List of tuples of embeddings
    tuple_people: list[tuple[np.ndarray, np.ndarray]] = []
    ground_truth: list[int] = [] # 0 = not the same person, 1 = same person
    
    # We add random embeddings from the selected people (so we have half the batch size embeddings from different people)
    while len(tuple_people) < half_batch_size:
        selected_person1: str = np.random.choice(people)
        selected_person2: str = np.random.choice(people)
        # TODO: Check if the tuple people already contains the selected people
        if selected_person1 != selected_person2:
            # Once we have selected two different people, we load their embeddings and pick one random embedding from each
            embeddings1: np.ndarray = load_npy(os.path.join(pathEmbeddings, selected_person1))
            if len(embeddings1) == 0:
                continue
            embeddings2: np.ndarray = load_npy(os.path.join(pathEmbeddings, selected_person2))
            if len(embeddings1) == 0:
                continue
            tuple_people.append((embeddings1, embeddings2))
            ground_truth.append(0)

    # We add random embeddings from the same person (so we have half the batch size embeddings from the same person)
    while len(tuple_people) < batch_size:
        selected_person: str = np.random.choice(people)
        embeddings: np.ndarray = load_npy(os.path.join(pathEmbeddings, selected_person))
        if len(embeddings) > 1:
            # Once we have selected two different people, we load their embeddings and pick one random embedding from each
            tuple_people.append((embeddings1, embeddings2))
            # TODO: Check if the new tuple have the same embeddings in the first and second position
            ground_truth.append(1)

    # Return numpy arrays
    tuple_people = np.array(tuple_people)
    ground_truth = np.array(ground_truth)
    return tuple_people, ground_truth


if __name__ == '__main__':
    extract_features()