import os
import numpy as np
import cv2

from insight_utilities.insight_interface import compareTwoFaces, get_face
from config import GALLERY_THRESHOLD, MAX_MISSING_FRAMES, NUM_BEST_FACES, NUMBER_OF_LAST_FACES, MATCH_MODALITY, MAX_GALLERY_SIZE


def build_gallery(other_environment: str, scenario_camera: str):
    """
    Build a gallery from the other environment.

    Args:
        other_environment (str): The other environment path.
        scenario_camera (str): The scenario and the camera.
    """
    # Load the gallery from the other environment and pick a random camera for a random scenario
    other_environment += "_faces"
    path = os.path.join(other_environment, os.listdir(other_environment)[0])
    # List of faces
    faces = os.listdir(path)
    # Create a new gallery
    gallery = {}
    # For each face in the gallery
    for face in faces:
        # Path of the face
        images = os.listdir(os.path.join(path, face))
        # Get all the .pgm images of the face
        images = list(filter(lambda x: x.endswith(".pgm"), images))[
            :MAX_GALLERY_SIZE]
        # Create a list of images
        gallery[face] = []
        # For each image
        for image in images:
            # Load the image
            img = cv2.imread(os.path.join(path, face, image))
            # Get features of the face
            face_feature, bboxes, kps = get_face(img)
            if face_feature is None:
                continue
            # Add the image to the gallery
            gallery[face].append(face_feature)
    # Return the gallery
    return gallery


def check_identity(gallery: dict, faces: list[np.ndarray]) -> dict[str, int]:
    """
    Check if the face is in the gallery.

    Args:
        gallery (dict): The gallery.
        faces (list[np.ndarray]): List of faces to check. 

    Returns:
        dict(str, int): Dictionary with names of subjects and the number of occurrences inside faces (only if > 0).
    """
    names: dict[str, int] = dict(
    )  # each entry is the number of occurrences of the corresponding name in faces
    # For each face
    for i, face in enumerate(faces):
        # Best name for the face
        best_name = ""
        # Best similarity
        best_sim = 0
        # For each subject in the gallery
        for subject in gallery:
            # For each face of the subject
            for face_feature in gallery[subject]:
                # Compare the face with the face in the gallery
                sim = compareTwoFaces(face, face_feature)
                # If the faces "are the same"
                if sim > best_sim:
                    # Update the best similarity and the best name
                    best_sim = sim
                    best_name = subject
        # So we have the best name for the face
        if best_sim > GALLERY_THRESHOLD:
            names[best_name] = names.get(best_name, 0) + 1
    # If the face is not in the gallery
    return names


class Identity:
    """
    Class that represents an identity.
    """
    last_id: int = 0

    def __init__(self, name: str = "Unknown"):
        self.id: int = Identity.last_id  # temporary id
        # definitive name of the identity, empty if the identity is not definitive
        self.name: str = name
        # bounding boxes of the faces in the frame
        self.bboxes: list[np.ndarray] = []
        self.kps: list[np.ndarray] = []  # keypoints of the faces in the frame
        # list of paths to the frames where the face is present, format: cam_frame
        self.frames: list[str] = []
        # list of features of the faces in the frame. The faces are alrady cropped
        self.faces: list[np.ndarray] = []
        self.last_faces: list[np.ndarray] = []
        self.missing_frames: int = 0  # number of frames where the face is not present
        # maximum number of frames where the face can be missing
        self.max_missing_frames: int = MAX_MISSING_FRAMES
        # Increment the last id
        Identity.last_id += 1

    def is_in_scene(self):
        """
        Check if the identity is in the scene.

        Returns:
            bool: True if the identity is in the scene, False otherwise.
        """
        return self.missing_frames < self.max_missing_frames

    def add_frame(self, face_features: np.ndarray, bboxes: np.ndarray, kps: np.ndarray, frame: str):
        """
        Add a frame to the identity.

        Args:
            frame (np.ndarray): The frame.
            bboxes (np.ndarray): The bounding boxes of the faces.
            kps (np.ndarray): The keypoints of the faces.
        """
        # Add the frame to the list of frames
        self.frames.append(frame)
        # Add the bounding boxes to the list of bounding boxes
        self.bboxes.append(bboxes)
        # Add the keypoints to the list of keypoints
        self.kps.append(kps)
        # Add the features to the list of features
        self.faces.append(face_features)

        self.last_faces.append(face_features)
        # If the list is too long, remove the oldest frame
        if len(self.last_faces) > NUMBER_OF_LAST_FACES:
            self.last_faces.pop(0)

    def match(self, face: np.ndarray):
        """
        Check if the face is the same as the one saved in self.last_faces.

        Args:
            face (np.ndarray): The face to check.

        Returns:
            float: The similarity between the face and the one saved in self.last_faces.
        """
        sim: float = 0

        if MATCH_MODALITY == "mean":
            for face_feature in self.last_faces:
                temp_sim = compareTwoFaces(face, face_feature)
                sim += temp_sim
            sim /= NUMBER_OF_LAST_FACES
        elif MATCH_MODALITY == "max":
            for face_feature in self.last_faces:
                temp_sim = compareTwoFaces(face, face_feature)
                sim = max(sim, temp_sim)
        # this method is going to be used to check if the face is the same as the one saved in self.last_faces

        return sim

    def get_biggest_faces(self):
        """
        Get the biggest faces in the identity.

        Returns:
            list[np.ndarray]: The biggest faces in the identity.
        """
        # Get the biggest faces
        biggest_faces: list[np.ndarray] = []
        bboxes_index: list[tuple[int, np.ndarray]] = enumerate(self.bboxes)
        # Sort the bounding boxes by the area of the bounding box
        biggest_bboxes: list[tuple[int, np.ndarray]] = sorted(
            bboxes_index, key=lambda x: abs(x[1][0] - x[1][2]) * abs(x[1][1] - x[1][3]), reverse=True)
        # Get the NUM_BEST_FACES biggest bounding boxes
        biggest_bboxes = biggest_bboxes[:NUM_BEST_FACES]
        # Get the faces corresponding to the biggest bounding boxes
        for bbox in biggest_bboxes:
            biggest_faces.append(self.faces[bbox[0]])
        # Return the biggest faces
        return biggest_faces

    def __repr__(self):
        return f"ID: {self.id}, Name: {self.name}, Frames: {self.frames}"
