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
        images = list(filter(lambda x: x.endswith(".pgm"), os.listdir(os.path.join(path, face))))[:max_size]
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
    """
    # Get the features of the face
    frame_features, bboxes, kpss = get_faces(frame)
    # If there are no faces
    if len(frame_features) == 0:
        return [], bboxes
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
    return names, bboxes