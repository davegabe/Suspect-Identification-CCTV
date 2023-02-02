import os
import os.path as osp
import numpy as np
import onnxruntime
from insight_utilities.scrfd import SCRFD                 #face detection
from insight_utilities.arcface_onnx import ArcFaceONNX    #face recognition

onnxruntime.set_default_logger_severity(3)

assets_dir = osp.expanduser('~/.insightface/models/buffalo_l')

detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
detector.prepare(0)
model_path = os.path.join(assets_dir, 'w600k_r50.onnx')
rec = ArcFaceONNX(model_path)
rec.prepare(0)

def get_faces(img: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Returns the faces from the image.

    Args:
        img (numpy.ndarray): An image.

    Returns:
        list[numpy.ndarray]: A list of faces.
        numpy.ndarray: A list of bounding boxes.
        numpy.ndarray: A list of keypoints.
    """
    bboxes, kpss = detector.autodetect(img)
    if bboxes.shape[0]==0:
        return [], [], []
    faces = []
    # For each detected face, get the face and append it to the faces list.
    for kps in kpss:
        face = rec.get(img, kps)
        faces.append(face)
    return faces, bboxes, kpss

def get_face(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the face from the image.

    Args:
        img (numpy.ndarray): An image.

    Returns:
        numpy.ndarray: A face.
        numpy.ndarray: A bounding box.
        numpy.ndarray: A keypoints.
    """
    bboxes, kpss = detector.autodetect(img, max_num=1)
    if bboxes.shape[0]==0:
        return None, [], []
    kps = kpss[0]
    face = rec.get(img, kps)
    return face, bboxes, kps


def compareTwoFaces(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """
    Compares two faces and returns a similarity score.

    Args:
        feat1 (numpy.ndarray): A face feature vector.
        feat2 (numpy.ndarray): A face feature vector.

    Returns:
        float: A similarity score.
    """
    sim = rec.compute_sim(feat1, feat2)     #this is a similarity score
    return sim
    