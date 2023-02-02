import numpy as np

from insight_utilities.insight_interface import get_faces

def more_faces_detected(frames: list[np.ndarray]) -> np.ndarray:
    """
    Select the frame with more faces detected

    Args:
        frames (list[np.ndarray]): List of frames from the three cameras
    
    Returns:
        np.ndarray: The frame with more faces detected
    """
    # Detect the faces inside each frame
    frame_faces = []
    for frame in frames:
        faces, bboxs, kpss = get_faces(frame)
        frame_faces.append(faces)
    # Return the frame with more faces detected
    best = frames[0]
    for i in range(1, len(frames)):
        if len(frame_faces[i]) > len(frame_faces[i-1]):
            best = frames[i]
    return best


def select_best_frames(frames: list[np.ndarray], criteria = "more_faces_detected") -> list[np.ndarray]:
    """
    Select the best frame from the three cameras

    Args:
        frames (list[np.ndarray]): List of frames from the three cameras
        criteria (str, optional): Criteria to select the best frame. Defaults to "more_faces_detected".
    
    Returns:
        np.ndarray: The best frames
    """
    return frames

    if len(frames) == 1:
        return frames[0]
    else:
        # IDEA: We could choose the frame with the highest confidence score
        # IDEA: We could choose the frame with more faces detected
        if criteria == "more_faces_detected":
            best = more_faces_detected(frames)
        # IDEA: We could use the faces extracted from the three cameras and check if they are the same person to continuously track the same person
        
        # ALSOIDEA: Otherwise we could just use position inside the frame (easier but not as good (and maybe not so easy))
        # ALSOALSOIDEA: Otherwise we could just use this approach (continuously track the same person) only if 1 camera is used
        return best
