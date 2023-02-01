import os
import numpy as np
import cv2

from insight_utilities.insight_interface import compareTwoFaces, get_face, get_faces
from config import MAX_MISSING_FRAMES, NUMBER_OF_LAST_FACES, MATCH_MODALITY, MAX_GALLERY_SIZE

def build_gallery(other_environment: str, scenario_camera: str):
    """
    Build a gallery from the other environment.

    Args:
        other_environment (str): The other environment path.
        max_size (int, optional): The maximum size of the gallery. Defaults to 100.
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
        # Get all the .pgm images of the face
        images = list(filter(lambda x: x.endswith(".pgm"),
                      os.listdir(os.path.join(path, face))))[:MAX_GALLERY_SIZE]
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
    names: dict[str, int] = dict() # each entry is the number of occurrences of the corresponding name in faces
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
                sim, _ = compareTwoFaces(face, face_feature)
                # If the faces "are the same"
                if sim > best_sim:
                    # Update the best similarity and the best name
                    best_sim = sim
                    best_name = subject
        # So we have the best name for the face
        names[best_name] = names.get(best_name, 0) + 1
    # If the face is not in the gallery
    return names


class Identity:
    """
    Class that represents an identity.
    """
    last_id: int = 0

    def __init__(self):
        self.id: int = Identity.last_id  # temporary id
        self.name: str = ""  # definitive name of the identity, empty if the identity is not definitive
        # bounding boxes of the faces in the frame
        self.bboxes: list[np.ndarray] = []
        self.kps: list[np.ndarray] = []  # keypoints of the faces in the frame
        # list of paths to the frames where the face is present
        self.frames: list[str] = []
        # list of features of the faces in the frame. The faces are alrady cropped
        self.faces: list[np.ndarray] = []
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

    def match(self, face: np.ndarray):
        assert MATCH_MODALITY in ["mean", "max"]
        sim = 0

        if MATCH_MODALITY == "mean":
            for face_feature in self.faces[:-NUMBER_OF_LAST_FACES]:
                temp_sim, _ = compareTwoFaces(face, face_feature)
                sim += temp_sim
            sim /= NUMBER_OF_LAST_FACES
        elif MATCH_MODALITY == "max":
            for face_feature in self.faces[:-NUMBER_OF_LAST_FACES]:
                    temp_sim, _ = compareTwoFaces(face, face_feature)
                    sim = max(sim, temp_sim)
        # this method is going to be used to check if the face is the same as the one saved in self.faces
        
        return sim


"""
temp_identities = {
    "identity_temp": {
        "frames": [frame1, frame2, frame3,  ...],     # each element of this (and also the following) list is a list of results for each camera (list of 3 elements)
        "bboxes": [bbox1, bbox2, bbox3, ...],
        "kpss": [kps1, kps2, kps3, ...],
        "features": [feature1, feature2, feature3, ...]
        "is_in_scene": True  # this will be False when the person disappears from the scene, at this point the decision module will do stuff and replace the temporary identity with a definitive one
    }
}
"""
