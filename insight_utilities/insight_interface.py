import os
import os.path as osp
import argparse
import cv2
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

def get_faces(img):
    """
    Returns the faces from the image.
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

def get_face(img):
    """
    input: an image
    output: a cropped image of the face in the image
    """
    bboxes, kpss = detector.autodetect(img, max_num=1)
    if bboxes.shape[0]==0:
        return None, [], []
    kps = kpss[0]
    face = rec.get(img, kps)
    return face, bboxes, kps


def compareTwoFaces(feat1, feat2): 
    """
    input: two images of faces. Note: this is not path but rather to the already read image.
    #img1 is the frame while img2 is the face from the database.

    output: a similarity score and a rapid conclusion, bounding boxes and keypoints of the probe image.
    if the similarity score is less than 0.2, the rapid conclusion is 0.
    if the similarity score is between 0.2 and 0.28, the rapid conclusion is 0.5.
    if the similarity score is greater than 0.28, the rapid conclusion is 1.
    if the rapid conclusion is -1.0, it means that no face was detected in the first image.
    if the rapid conclusion is -2.0, it means that no face was detected in the second image.
    """
    sim = rec.compute_sim(feat1, feat2)     #this is a similarity score
    if sim<0.2:
        rapid_conclusion = 0
    elif sim>=0.2 and sim<0.28:
        rapid_conclusion = 0.5
    else:
        rapid_conclusion = 1
    #also return the bounding boxes and keypoints of the probe image
    return sim, rapid_conclusion
    