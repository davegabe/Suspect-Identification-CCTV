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
    Returns a random embedding from a person (loaded from file).
    """
    # Load the numpy array
    array = np.load(path, allow_pickle=True)[0]
    # Check if the array is empty
    if len(array) == 0:
        return np.array([])
    # Get random embedding
    data: dict = array[np.random.randint(0, len(array))]
    # Return embedding
    embedding = data["embedding"]
    return embedding

def load_embeddings(batch_size=32, is_train=True, partition=0.8) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads two people embeddings and returns them.
    """
    # Get all people
    people = list(filter(lambda x: x.endswith('.npy'), os.listdir(pathEmbeddings)))
    if (is_train):
        people = people[:int(len(people) * partition)]
    else:
        people = people[int(len(people) * partition):]
    # Get half the batch size
    half_batch_size = batch_size // 2
    # List of tuples concatenated embeddings of two people
    tuple_people: list[np.ndarray] = []
    ground_truth: list[int] = [] # 0 = not the same person, 1 = same person
    
    # We add random embeddings from the selected people (so we have half the batch size embeddings from different people)
    while len(tuple_people) < half_batch_size:
        selected_person1: str = np.random.choice(people)
        embeddings1: np.ndarray = load_npy(os.path.join(pathEmbeddings, selected_person1))
        selected_person2: str = np.random.choice(people)
        embeddings2: np.ndarray = load_npy(os.path.join(pathEmbeddings, selected_person2))
        # TODO: Check if the tuple people already contains the selected people
        if selected_person1 != selected_person2 and len(embeddings1) > 0 and len(embeddings2) > 0:
            # Once we have selected two different people, we load their embeddings and pick one random embedding from each
            tuple_people.append(np.concatenate((embeddings1, embeddings2)))
            ground_truth.append(0)

    # We add random embeddings from the same person (so we have half the batch size embeddings from the same person)
    while len(tuple_people) < batch_size:
        selected_person: str = np.random.choice(people)
        embeddings1: np.ndarray = load_npy(os.path.join(pathEmbeddings, selected_person))
        if len(embeddings1) > 1:
            # Once we have selected two different people, we load their embeddings and pick one random embedding from each
            embeddings2: np.ndarray = load_npy(os.path.join(pathEmbeddings, selected_person))
            # TODO: Check if the new tuple have the same embeddings in the first and second position
            tuple_people.append(np.concatenate((embeddings1, embeddings2)))
            ground_truth.append(1)

    # Return numpy arrays
    tuple_people = np.array(tuple_people)
    ground_truth = np.array(ground_truth)
    return tuple_people, ground_truth


if __name__ == '__main__':
    extract_features()