import os
import numpy as np
import xml.etree.ElementTree as ET
from modules.gallery_module import Identity

class GroundTruthItem():
    """
    Class that represents a ground truth item. There is one for each identity in each frame.
    """
    def __init__(self, name: str, left_eye: tuple[int, int], right_eye: tuple[int, int]):
        """
        Args:
            name: Name of the person.
            left_eye: Left eye (x, y).
            right_eye: Right eye (x, y).
        """
        self.name = name
        self.left_eye = left_eye
        self.right_eye = right_eye
        

    def __repr__(self) -> str:
        return f"Id: {self.name}, Left eye: {self.left_eye}, Right eye: {self.right_eye}"

class GroundTruthIdentity():
    """
    Class that represents an identity in the ground truth.
    """
    def __init__(self, name: str, pg: int):
        """
        Args:
            name: Name of the person.
        """
        self.name = name
        self.rank_postitions: dict[int, int] = dict() # Dict where each key corresponds to the number of times it was ranked in that position.
        self.pg = pg # Number of frames where the person was present

    def compute_dir(self, rank: int):
        """
        Detection and Identification Rate (DIR) is the ratio of the number of times the system correctly identifies the person
        in a rank position minor than rank.
        """
        # Get the number of times the person was ranked in a position minor than rank
        n = sum([self.rank_postitions[i] for i in range(rank)])
        # Return the DIR
        return n / self.pg

        

class EvalIdentityItem():
    """
    Class that represents an identity item.
    """
    def __init__(self, ranks: str, bbox: np.ndarray):
        """
        Args:
            ranks: Possible identities for the face ordered by rank.
            bbox: Bounding box.
        """
        self.ranks = ranks
        self.bbox = bbox

def build_groundtruth(groundtruth_paths: list[str]) -> tuple[dict[str, list[GroundTruthItem]], list[GroundTruthIdentity]]:
    """
    Builds a dictionary of groundtruth faces.

    Args:
        groundtruth_paths: List of paths to the groundtruth faces.

    Returns:
        A dictionary of groundtruth faces where the key is the frame number and the value is a list of GroundTruthItem.
    """
    groundtruth: dict[str, list[GroundTruthItem]] = {}
    name_occurrences: dict[str, int] = {}
    for groundtruth_path in groundtruth_paths:
        cam_n = os.path.basename(groundtruth_path).split("_")[-1].split(".")[0][1] # Assumes the name of the file is "ENV_SN_CN.n.xml"
        tree = ET.parse(groundtruth_path)
        for frame in tree.getroot():
            frame_n = frame.attrib["number"]
            groundtruth[f"{cam_n}_{frame_n}"] = []
            for person in frame.findall("person"):
                r_eye = person.find("rightEye")
                l_eye = person.find("leftEye")
                groundtruth[f"{cam_n}_{frame_n}"].append(
                    GroundTruthItem(
                        person.attrib["id"], 
                        (int(l_eye.attrib["x"]), int(l_eye.attrib["y"])), 
                        (int(r_eye.attrib["x"]), int(r_eye.attrib["y"]))
                    )
                )
                name_occurrences[person.attrib["id"]] = name_occurrences.get(person.attrib["id"], 0) + 1
    groundtruth_identities = [GroundTruthIdentity(name, occurrences) for name, occurrences in name_occurrences.items()]
    return groundtruth, groundtruth_identities

def list_to_dict_identities(identities: list[Identity]) -> dict[str, list[EvalIdentityItem]]:
    """
    Converts a list of identities to a dictionary of identities.
    
    Args:
        identities: List of identities.

    Returns:
        A dictionary of identities where the key is the frame number and the value is a list of EvalIdentityItem.
    """
    # Initialize the dictionary
    identities_dict: dict[str, list[EvalIdentityItem]] = {}
    # For each identity, add the frame number and the bounding box in the frame and the ranks
    for identity in identities:
        # For each frame, add the bounding box of the face and the ranks
        for i, frame in enumerate(identity.frames):
            identities_dict[frame] = identities_dict.get(frame, []) + [EvalIdentityItem(identity.ranks, identity.bboxes[i])]
    # Return the dictionary
    return identities_dict

def contains_eyes(bbox: np.ndarray, left_eye: np.ndarray, right_eye: np.ndarray) -> bool:
    """
    Checks if the eye is inside the bounding box.

    Args:
        bbox: Bounding box.
        left_eye: Left eye (x, y)
        right_eye: Right eye (x, y)

    Returns:
        True if the eye is inside the bounding box, False otherwise.
    """
    # Get the coordinates of the bounding box
    x, y, w, h = bbox
    # Get the coordinates of the eyes
    x_l, y_l = left_eye
    x_r, y_r = right_eye
    # Check if the left eye is inside the bounding box
    is_left_inside = x <= x_l <= x + w and y <= y_l <= y + h
    # Check if the right eye is inside the bounding box
    is_right_inside = x <= x_r <= x + w and y <= y_r <= y + h
    # Check if the eyes are inside the bounding box
    return is_left_inside and is_right_inside

def evaluate_system(known_identities: list[Identity], unknown_identities: list[Identity], groundtruth_paths: list[str], all_frames: list[str]):
    """
    Evaluates the system.

    Args:
        known_identities: List of known identities.
        unknown_identities: List of unknown identities.
        groundtruth_faces_path: Path to the groundtruth faces.
    """
    # Get the groundtruth faces
    groundtruth, groundtruth_identities = build_groundtruth(groundtruth_paths)
    # Create detected dict
    detected_known = list_to_dict_identities(known_identities)
    detected_unknown = list_to_dict_identities(unknown_identities)
    # Get the frames where the system detected faces
    detected_known_frames = list(detected_known.keys())
    detected_unknown_frames = list(detected_unknown.keys())

    incorrect_frames = set() # the frames where the system detected faces but didn't detect the correct person
    genuine_rejections: dict[str, list[str]] = {} # detected unknown faces (negative identification) that are not in the gallery
    false_acceptances: dict[str, list[str]] = {} # detected known faces that are not in the gallery
    false_rejections: dict[str, list[str]] = {} # detected known faces that are in in the gallery, but not the actual ones (they are different from ground thruth), AND detected unknown faces that are in the gallery, AND not detected faces that are in the gallery

    # For each frame
    for frame in all_frames:
        # Get identities detected in the frame
        known_ids = detected_known.get(frame, [])
        unknown_ids = detected_unknown.get(frame, [])
        # Get the groundtruth identities in the frame
        groundtruth_ids = groundtruth.get(frame, [])

        # For each known identity
        for known_id in known_ids:
            # We find the groundtruth identity corresponding to the known identity (check if the eyes are inside the bounding box)
            groundtruth_id: GroundTruthItem = next((x for x in groundtruth_ids if contains_eyes(known_id.bbox, x.left_eye, x.right_eye)), None)
            # If there was no face there, it's a false rejection OR 
            # if the face was there, but its similarity to the gallery is too low (doesn't appear in ranks), it's a false rejection
            if groundtruth_id is None:
                # We have not correctly identified the person
                false_acceptances[frame] = false_acceptances.get(frame, []) + [known_id]
                incorrect_frames.add(frame)
            elif groundtruth_id.name not in known_id.ranks:
                false_rejections[frame] = false_rejections.get(frame, []) + [known_id]
                incorrect_frames.add(frame)
            # If the face was there and it's the correct person, we have correctly identified the person
            elif groundtruth_id.name in known_id.ranks:
                # We get the rank of the correct person (where it appears in the ranks list)
                rank = known_id.ranks.index(groundtruth_id.name)
                # If the rank is not 0, it's a false rejection
                if rank != 0:
                    false_rejections[frame] = false_rejections.get(frame, []) + [known_id]
                    incorrect_frames.add(frame)
                # Get the identity from the groundtruth identities with the same name
                groundtruth_identity = next((x for x in groundtruth_identities if x.name == groundtruth_id.name), None)
                assert groundtruth_identity is not None, f"Groundtruth identity {groundtruth_id.name} not found!"
                # Get the rank of the groundtruth identity
                groundtruth_identity.rank_postitions[rank] = groundtruth_identity.rank_postitions.get(rank, 0) + 1

        # For each identity in groundtruth
        for groundtruth_id in groundtruth_ids:
            # Check if the identity is in the known identities
            found = map(lambda known_id: groundtruth_id.name in known_id.ranks and contains_eyes(known_id.bbox, groundtruth_id.left_eye, groundtruth_id.right_eye), known_ids)
            # If there is a face in the groundtruth that is not in the detected faces, it's a false rejection
            if not any(found):
                false_rejections[frame] = false_rejections.get(frame, []) + [groundtruth_id]
                incorrect_frames.add(frame)

        # for each unknown identity if its present as unknown in the groundtruth, it's a genuine rejection
        unknown_identities_in_groundtruth = [x for x in groundtruth_ids if x.name == "Unknown"]
        for i, unknown_id in enumerate(unknown_ids):
            if i < len(unknown_identities_in_groundtruth):
                genuine_rejections[frame] = genuine_rejections.get(frame, []) + [unknown_id.ranks[0]] # ranks[0] contains "Unknown"
                incorrect_frames.add(frame)


    # TODO: ricalcolare roba in base a false negatives e false positives, etc            
    # Get the frames where the system detected faces and detected the correct person
    correct_frames = set(detected_known_frames + detected_unknown_frames) - incorrect_frames
    # Calculate the precision
    precision = len(correct_frames) / len(detected_known_frames + detected_unknown_frames)
    # Calculate the recall
    recall = len(correct_frames) / len(groundtruth.keys())
    # Calculate the F1 score
    f1_score = 2 * precision * recall / (precision + recall)
    # Print the results

    #false acceptance rate = false positives / (false positives + true negatives)
    #false rejection rate = false negatives / (false negatives + true positives)
    #false acceptance rate = false positives / (false positives + true negatives)
    #false rejection rate = false negatives / (false negatives + true positives)

    #ROC


    """
    Precision:  True positive / (True positive + False positive)
    Recall: True positive / (True positive + False negative)
    F1 score: 2 * Precision * Recall / (Precision + Recall)
    
    FRR: False negative / (False negative + True positive)
    FAR: False positive / (False positive + True negative)

    CMS(at rank k): True positive / (True positive + False positive + False negative)       #probability of correct match at rank k
    CMC: cumulative match characteristic

    EER(t): equal error rate at threshold t = {x: FRR(t)=x and FAR(t)=x }

    ROC and DET: Receiver Operating Characteristic and Detection Error Tradeoff    
    """

    FP = len([y for sublist in false_positives.values() for y in sublist])
    FN = len([y for sublist in false_negatives.values() for y in sublist])
    TP = len([y for sublist in true_positives.values() for y in sublist])
    TN = len([y for sublist in true_negatives.values() for y in sublist])

    tot = 0
    for i in true_positives.values():
        tot += len(i)
    print(f"TP: {TP}, oppure: {tot}")
    FRR = FN / (FN + TP)
    FAR = FP / (FP + TN)
    CMS = TP / (TP + FP + FN) # at rank 1

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 score: ", f1_score)
    print(f"False positives: {FP}, False negatives: {FN}")
    print(f"True positives: {TP}, True negatives: {TN}")
    print(f"False acceptance rate: {FAR}, False rejection rate: {FRR}")
    print(f"CMS: {CMS}")
    print("Incorrect frames: ", len(incorrect_frames))

