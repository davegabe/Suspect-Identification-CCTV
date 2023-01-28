import tensorflow as tf
import os
import numpy as np

class CustomDataLoader(tf.keras.utils.Sequence):
    def __init__(self, batch_size, input_size=1024, is_train=True, train_partition=1, embedding_path="./embeddings/"):
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = False
        self.is_train = is_train
        self.train_partition = train_partition
        # Get the list of people
        self.people = list(filter(lambda x: x.endswith('.npy'), os.listdir(embedding_path)))
        self.people_more_embeddings = [] # People with more than one embedding
        self.n = len(self.people)
        # Load the embeddings
        self.embeddings = dict()
        for person in self.people:
            self.embeddings[person] = self.load_embeddings_from_npy(os.path.join(embedding_path, person))
            if len(self.embeddings[person]) > 1:
                self.people_more_embeddings.append(person)
            elif len(self.embeddings[person]) == 0:
                self.people.remove(person)
                self.embeddings.pop(person)
        # Split the people into train and test
        self.people = list(self.embeddings.keys())
        self.train_people = self.people[:int(len(self.people) * self.train_partition)]
        self.test_people = self.people[int(len(self.people) * self.train_partition):]
        # Get people with more than one embedding
        self.train_people_more_embeddings = list(filter(lambda x: x in self.people_more_embeddings, self.train_people))
        self.test_people_more_embeddings = list(filter(lambda x: x in self.people_more_embeddings, self.test_people))

    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Returns the euclidean distance between two embeddings.
        Resulting shape will be the same as the input.
        """
        return np.sqrt(np.square(x1 - x2))

    def load_embeddings_from_npy(self, path: str) ->  list[np.ndarray]:
        """
        Loads the embeddings from a person (loaded from file).
        """
        # Load the numpy array
        try:
            array = np.load(path, allow_pickle=True)[0]
            array = list(map(lambda x: x["embedding"], array))
        except:
            array = []
        return array

    def random_embedding(self, embeddings: list[np.ndarray]) -> np.ndarray:
        """
        Returns a random embedding from a person (loaded from file).
        """
        # Get random embedding
        data: dict = embeddings[np.random.randint(0, len(embeddings))]
        # Return embedding
        embedding = data["embedding"]
        return embedding

    def random_embedding_from_npy(self, path: str) -> np.ndarray:
        """
        Returns a random embedding from a person (loaded from file).
        """
        # Load the numpy array
        array = self.load_embeddings_from_npy(path)
        # Check if the array is empty
        if len(array) > 0:
            # Get random embedding
            data: dict = array[np.random.randint(0, len(array))]
            # Return embedding
            embedding = data["embedding"]
            return embedding
        else:
            raise Exception("The array is empty")

    def load_embeddings(self, batch_size=32, is_train=True) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads two people embeddings and returns them.
        """
        # Define the set of people to use
        if is_train:
            people = self.train_people
            people_more_embeddings = self.train_people_more_embeddings
        else:
            people = self.test_people
            people_more_embeddings = self.test_people_more_embeddings
        # Get half the batch size
        half_batch_size = batch_size // 2
        # List of euclidean distances
        euclidean_distances: list[np.ndarray] = []
        ground_truth: list[int] = [] # 0 = not the same person, 1 = same person
        
        # We add random embeddings from the selected people (so we have half the batch size embeddings from different people)
        while len(euclidean_distances) < half_batch_size:
            # We select two different people, we load their embeddings and pick one random embedding from each
            selected_people: str = np.random.choice(people, size=2, replace=False)
            # TODO: Check if the tuple people already contains the selected people
            emb1 = self.embeddings[selected_people[0]]
            selected_emb1: np.ndarray = emb1[np.random.randint(0, len(emb1))]
            emb2 = self.embeddings[selected_people[1]]
            selected_emb2: np.ndarray = emb2[np.random.randint(0, len(emb2))]
            euclidean_distances.append(self.euclidean_distance(selected_emb1, selected_emb2))
            ground_truth.append(0)

        # We add random embeddings from the same person (so we have half the batch size embeddings from the same person)
        while len(euclidean_distances) < batch_size:
            selected_person: str = np.random.choice(people_more_embeddings)
            emb = self.embeddings[selected_person]
            indexes = np.random.randint(0, len(emb), size=2)
            euclidean_distances.append(self.euclidean_distance(emb[indexes[0]], emb[indexes[1]]))
            ground_truth.append(1)

        # Return numpy arrays
        euclidean_distances = np.array(euclidean_distances, dtype=np.float)
        ground_truth = np.array(ground_truth)
        return euclidean_distances, ground_truth

    def __getitem__(self, index):
        X, y = self.load_embeddings(self.batch_size, self.is_train)
        return X, y

    def __len__(self):
        return self.n // self.batch_size
