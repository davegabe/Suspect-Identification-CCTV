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
        return f"(Id: {self.name}, Left eye: {self.left_eye}, Right eye: {self.right_eye})"

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
        self.is_impostor = False # True if the person is an impostor, False otherwise

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
        self.ranks = ranks # TODO: Rename to ranked_names
        self.bbox = bbox

def build_groundtruth(groundtruth_paths: list[str], gallery: dict[str, list[np.ndarray]]) -> tuple[dict[str, list[GroundTruthItem]], list[GroundTruthIdentity]]:
    """
    Builds a dictionary of groundtruth faces.

    Args:
        groundtruth_paths: List of paths to the groundtruth faces.

    Returns:
        A dictionary of groundtruth faces where the key is the frame number and the value is a list of GroundTruthItem.
        A list where each item represents an identity in the groundtruth.
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
    # Check if the person is an impostor
    for identity in groundtruth_identities:
        if identity.name not in gallery.keys():
            identity.is_impostor = True
        if identity.name == "Unknown": # all Unknowns are associated to a single identity (in theory)
            identity.is_impostor = True
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
            identities_dict[frame] = identities_dict.get(frame, []) + [EvalIdentityItem(identity.ranked_names, identity.bboxes[i])]
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
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    # Get the coordinates of the eyes
    x_l, y_l = left_eye
    x_r, y_r = right_eye
    # Check if the left eye is inside the bounding box
    is_left_inside = x1 <= x_l <= x2 and y1 <= y_l <= y2
    # Check if the right eye is inside the bounding box
    is_right_inside = x1 <= x_r <= x2 and y1 <= y_r <= y2
    # Check if the eyes are inside the bounding box
    return is_left_inside and is_right_inside

def evaluate_system(known_identities: list[Identity], unknown_identities: list[Identity], groundtruth_paths: list[str], all_frames: list[str], gallery: dict[str, list[np.ndarray]]):
    """
    Evaluates the system.

    Args:
        known_identities: List of known identities.
        unknown_identities: List of unknown identities.
        groundtruth_faces_path: Path to the groundtruth faces.
    """
    # Get the groundtruth faces
    groundtruth, groundtruth_identities = build_groundtruth(groundtruth_paths, gallery)
    
    n_impostor_faces = 0
    n_genuine_faces = 0
    for frame in groundtruth.keys():
        for face in groundtruth[frame]:
            identity = next((i for i in groundtruth_identities if face.name == i.name))
            if identity.is_impostor:
                n_impostor_faces += 1
            else:
                n_genuine_faces += 1
    # Create detected dict
    detected_known = list_to_dict_identities(known_identities)
    detected_unknown = list_to_dict_identities(unknown_identities)

    incorrect_frames = set() # the frames where the system detected faces but didn't detect the correct person
    genuine_rejections: dict[str, list[str]] = {} # detected unknown faces (negative identification) that are not in the gallery
    false_acceptances: dict[str, list[str]] = {} # detected known faces that are not in the gallery
    false_rejections: dict[str, list[str]] = {} # detected known faces that are in in the gallery, but not the actual ones (they are different from ground thruth), AND detected unknown faces that are in the gallery, AND not detected faces that are in the gallery

    """
    Assumiamo che la detection sia andata a buon fine e che dobbiamo solo valutare la identification
    In questo contesto, per galleria, intendiamo le persone che sono state riconosciute come conosciute, mentre per impostori quelle che sono state riconosciute come sconosciute.
    In ogni frame:
        prendo le facce nella groundtruth (galleria+impostori)
        controllo che tutti quelli nella galleria siano nei known, e tutti quelli degli impostori siano nei unknown
        per galleria:
            se non ci sono alcuni nei known: false reject
            se ci sono ma sono non a rank 1: false reject+calcola DIR
            tutti gli altri: genuine accept
        per impostor:
            se non ci sono alcuni negli unknown: boh (non fare niente?) (dovrebbe essere coperto da quelli che sono tra i known ma non in galleria)
            tutti gli altri: genuine reject
        
        poi per ogni known:
            se non e' nella galleria: false accept
        per ogni unknown:
            se non e' negli impostori: boh (non fare niente?) (dovrebbe essere coperto da quelli in galleria che non sono tra i known)
    """
    debug_contained = 0
    # For each frame
    for frame in all_frames:
        frame = frame.split(".")[0] # remove the extension
        # print(f"Evaluating Frame {frame}")
        # Get identities detected in the frame
        known_ids = detected_known.get(frame, [])
        unknown_ids = detected_unknown.get(frame, [])
        # Get the groundtruth identities in the frame
        groundtruth_its = groundtruth.get(frame, [])

        for git in groundtruth_its:
            try:
                real_id = next((i for i in groundtruth_identities if i.name == git.name))
            except StopIteration:
                print(f"Identity {git.name} not found in the groundtruth identities")
                exit(1)
            if real_id.is_impostor: # Handle impostors
                for uid in unknown_ids:
                    if contains_eyes(uid.bbox, git.left_eye, git.right_eye):
                        debug_contained += 1
                        genuine_rejections[frame] = genuine_rejections.get(frame, []) + [uid]
                        break
                    # If it isn't in unknown, it should be in known and would be a false accept, otherwise it was not detected and it would be fine as well
            else: # Handle genuine
                in_known = False
                for kid in known_ids:
                    # We take the corresponding face
                    if contains_eyes(kid.bbox, git.left_eye, git.right_eye):
                        debug_contained += 1
                        in_known = True # Face is found
                        try:
                            rank = kid.ranks.index(git.name)
                        except ValueError: # Face is recognized but real identity is below the threshold
                            false_rejections[frame] = false_rejections.get(frame, []) + [kid]
                            break # We found face associated to eyes coordinates of ground identity, so we can skip to next ground identity
                        if rank != 0: # Face is recognized but not at rank 1
                            false_rejections[frame] = false_rejections.get(frame, []) + [kid]
                        # At this point we know face is recognized and identity is at some rank, so we can use it in DIR
                        real_id.rank_postitions[rank] = real_id.rank_postitions.get(rank, 0) + 1
                        break # We found face associated to eyes coordinates of ground identity, so we can skip to next ground identity 
                if not in_known: # If face is not in known we hope it is in unknown, and it is a false rejection (even if it is not in unknown it is a false rejection, but in that case face was not even detected)
                    false_rejections[frame] = false_rejections.get(frame, []) + [kid]
                    
        # Check for false acceptances (known ids that are not in the gallery)
        for kid in known_ids:
            if not any(contains_eyes(kid.bbox, x.left_eye, x.right_eye) for x in groundtruth_its): # should also check if x is impostor?
                debug_contained += 1
                false_acceptances[frame] = false_acceptances.get(frame, []) + [kid]

    print(f"Genuine Rejections: {len(genuine_rejections)}")
    print(f"False Rejections: {len(false_rejections)}")
    print(f"False Acceptances: {len(false_acceptances)}")
    print(f"People in gallery: {len(list(filter(lambda x:not x.is_impostor, groundtruth_identities)))}")
    # FAR = FA / tutti gli impostori = FA / (FA+GR)
    far = sum(map(len, false_acceptances.values())) / (n_impostor_faces)
    # DIR(k) = tutti quelli a rank <= k / tutti quelli che dovrebbero essere a rank 1
    dir = {}
    # Get the max rank between all identities
    max_k = 0
    for i in filter(lambda x: not x.is_impostor, groundtruth_identities):
        if len(i.rank_postitions.keys()) == 0:
            continue
        tmax = max(list(i.rank_postitions.keys()))
        if tmax > max_k:
            max_k = tmax
    print(f"Max rank: {max_k}")
    for k in range(max_k+1):
        total_rank = 0
        for i in range(k+1):
            total_rank += sum(map(lambda x: x.rank_postitions.get(i, 0), groundtruth_identities))
        dir[k] = total_rank / n_genuine_faces
    # FRR = FR / tutti i genuini = FR / (FR+FA)
    frr =  len(false_rejections) / n_genuine_faces

    total_for_each_rank = {}
    for i in groundtruth_identities:
        for k, v in i.rank_postitions.items():
            total_for_each_rank[k] = total_for_each_rank.get(k, 0) + v
    print(f"Total for each rank: {total_for_each_rank}")

    all_faces = 0
    for f in groundtruth:
        for face in groundtruth[f]:
            all_faces += 1
    print(f"Genuine faces: {n_genuine_faces}, impostor faces: {n_impostor_faces}, all faces: {all_faces}")
    print(f"Debug contained: {debug_contained}")

    # Print results
    print(f"False Acceptance Rate: {far}")
    print(f"False Rejection Rate: {frr}")
    for k, v in dir.items():
        print(f"Detection rate in Rank {k}: {v}")
    # print(f"Groundtruth identities: {list(map(lambda i: str(i.is_impostor) + i.name, groundtruth_identities))}")

