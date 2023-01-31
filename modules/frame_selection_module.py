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


def select_best_frame(frames: list[np.ndarray], criteria = "more_faces_detected") -> np.ndarray:
    """
    Select the best frame from the three cameras

    Args:
        frames (list[np.ndarray]): List of frames from the three cameras
        criteria (str, optional): Criteria to select the best frame. Defaults to "more_faces_detected".
    
    Returns:
        np.ndarray: The best frame
    """
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
    
# where fram-i = tuple(frames[0], faces[0], bboxs[0], kpss[0])
# Result of frame selection:
# Each element of the list is a list of "fram-i" tuples having some common characteristics and so we have to take the same decision
# [[fram1],[fram2]] frame selection banale dove ogni frame avrà una decisione indipendente
# [[fram1,fram2,fram3,fram4,fram5], ...] frame selection dove ogni frame sarà ad esempio sulla stessa tracked person e quindi si dovrà prendere la stessa decisione. Quindi fram1,fram2,fram3,fram4,fram5 sono tutti sulla stessa persona e quindi si dovrà prendere la stessa decisione per tutti
