import os
import numpy as np
import cv2

from insight_utilities.insight_interface import compareTwoFaces, get_face, get_faces


def build_gallery(other_environment: str, scenario_camera: str, max_size=100):
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
                      os.listdir(os.path.join(path, face))))[:max_size]
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


def check_identity(gallery: dict, frame: np.ndarray):
    """
    Check if the face is in the gallery.

    Args:
        gallery (dict): The gallery.
        frame (np.ndarray): The frame from the camera.

    Returns:
        str: The name of the face if it is in the gallery, "Unknown" otherwise, None if there is no face.
        [np.ndarray]: The bounding boxes of the faces.
        [np.ndarray]: The keypoints of the faces.
    """
    # Get the features of the face
    frame_features, bboxes, kpss = get_faces(frame)
    # If there are no faces
    if len(frame_features) == 0:
        return [], bboxes, kpss
    # For each face in the frame
    names = []
    for i, frame_feature in enumerate(frame_features):
        # For each face in the gallery
        for face in gallery:
            # If the face has already been found
            if len(names) == i+1:
                break
            # For every image of the face
            for face_feature in gallery[face]:
                # Check if the face is in the gallery
                _, is_same = compareTwoFaces(frame_feature, face_feature)
                if is_same == 1:
                    names.append(face)
                    break
        # If the face is not in the gallery
        if len(names) == i:
            names.append("Unknown")
    # If the face is not in the gallery
    return names, bboxes, kpss


class Identity:
    """
    Class that represents an identity.
    """
    last_id: int = 0

    def __init__(self, max_missing_frames: int = 10):
        self.id: int = last_id  # temporary id
        self.final_id: str = ""  # definitive id, empty if the identity is not definitive
        # bounding boxes of the faces in the frame
        self.bboxes: list[np.ndarray] = []
        self.kps: list[np.ndarray] = []  # keypoints of the faces in the frame
        # list of paths to the frames where the face is present
        self.frames: list[str] = []
        # list of features of the faces in the frame. The faces are alrady cropped
        self.faces: list[np.ndarray] = []
        self.missing_frames: int = 0  # number of frames where the face is not present
        # maximum number of frames where the face can be missing
        self.max_missing_frames: int = max_missing_frames
        # Increment the last id
        last_id += 1

    def is_in_scene(self):
        """
        Check if the identity is in the scene.

        Returns:
            bool: True if the identity is in the scene, False otherwise.
        """
        return self.missing_frames < self.max_missing_frames

    def add_frame(self, frame: np.ndarray, bboxes: np.ndarray, kps: np.ndarray, face_features: np.ndarray):
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

    def check_if_identity_matches(self, face):
        # this method is going to be used to check if the face is the same as the one saved in self.faces
        for face_feature in self.faces[:-5]:
            _, is_same = compareTwoFaces(face, face_feature)
            if is_same == 1:
                return True
        return False


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
