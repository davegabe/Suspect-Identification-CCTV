import os
import numpy as np
import cv2

from insight_utilities.insight_interface import compareTwoFaces

def build_gallery(other_environment: str, max_size=100):
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
        images = os.listdir(os.path.join(path, face))[:max_size]
        # Create a list of images
        gallery[face] = []
        # For each image
        for image in images:
            # Load the image
            img = cv2.imread(os.path.join(path, face, image))
            # Add the image to the gallery
            gallery[face].append(img)
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
    # For each face in the gallery
    for face in gallery:
        # For every image of the face
        for image in gallery[face]:
            # Check if the face is in the gallery
            _, is_same, boundingBoxes, keyPoints = compareTwoFaces(frame, image)
        if is_same == 1:
            return face, boundingBoxes, keyPoints
        if is_same < 0:
            # Return None
            return None, [], []
    # If the face is not in the gallery
    return "Unknown", boundingBoxes, keyPoints