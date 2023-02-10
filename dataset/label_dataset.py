import os
import cv2
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm
import numpy as np

from insight_utilities.insight_interface import get_faces
from modules.evaluation_module import contains_eyes
from config import TEST_PATH, TEST_SCENARIO, TEST_SCENARIO2, MAX_CAMERAS, GROUNDTRUTH_PATH

class UnknownFace:
    """
    Unknown face class
    """
    def __init__(self, camera: str, frame: str, left_eye: tuple[int, int], right_eye: tuple[int, int]):
        self.frame = frame
        self.camera = camera
        self.left_eye = left_eye
        self.right_eye = right_eye

def main():
    data_path = "data/"
    protocols = {
    #"P1E": ["S1", "S2", "S3", "S4"],
    #"P1L": ["S1", "S2", "S3", "S4"],
    #"P2E": ["S1", "S2", "S3", "S4"],
    "P2E": ["S3", "S4"],
    "P2L": ["S1", "S2", "S3", "S4"],
    }

    scenarios: list[tuple[str, str]] = []
    for environment in protocols: # P1E, P1L, P2E, P2L
        env_scenarios = protocols[environment]
        # We append to scenarios the corresponding folders to env_scenarios
        for scenario in env_scenarios: # S1, S2, S3, S4
            path_env_scenario = f"{environment}_{scenario}"
            if environment in ["P2E", "P2L"]:
                scenarios.append((environment, path_env_scenario, ".1"))
                scenarios.append((environment, path_env_scenario, ".2"))
            else:
                scenarios.append((environment, path_env_scenario, ""))

    # For each camera find the unknown faces
    for environment, scenario, suffix_scenario in scenarios:
        # Clear the temp_unknown_faces folder
        dest_path = os.path.join("temp_unknown_faces", scenario+suffix_scenario)
        shutil.rmtree(dest_path, ignore_errors=True)
        os.makedirs(dest_path, exist_ok=True)

        for camera in range(MAX_CAMERAS):
            # Make dirs
            os.makedirs(os.path.join(dest_path, f"C{camera+1}"), exist_ok=True)
            path = os.path.join(data_path, environment, f"{scenario}_C{camera+1}{suffix_scenario}") # path of the camera

            # Load the groundtruth
            groundtruth_file = os.path.join(GROUNDTRUTH_PATH, f"{scenario}_C{camera+1}{suffix_scenario}.xml")
            tree = ET.parse(groundtruth_file)
            
            # For each frame in the groundtruth, autodetect images and check if there are unknown faces
            print(f"Detecting unknown faces in {scenario}_C{camera+1}{suffix_scenario}...")
            cam_n = os.path.basename(groundtruth_file).split("_")[-1].split(".")[0][1] # Assumes the name of the file is "ENV_SN_CN.n.xml"
            for frame in tqdm(tree.getroot()):
                frame_n = frame.attrib["number"]
                # Get the frame image
                img = cv2.imread(os.path.join(path, f"{frame_n}.jpg"))
                # Get the faces from the frame
                faces, bboxes, kpss = get_faces(img)
                if len(faces) == 0:
                    continue
                # For each person in the groundtruth, remove the face from the list of faces
                for person in frame:
                    # Get the left eye position
                    left_eye = person.find("leftEye")
                    if left_eye is None:
                        continue
                    left_eye_pos = (int(left_eye.attrib["x"]), int(left_eye.attrib["y"]))
                    # Get the right eye position
                    right_eye = person.find("rightEye")
                    if right_eye is None:
                        continue
                    right_eye_pos = (int(right_eye.attrib["x"]), int(right_eye.attrib["y"]))
                    # Find the bbox of the corresponding face
                    index = -1
                    for i, bbox in enumerate(bboxes):
                        if contains_eyes(bbox, left_eye_pos, right_eye_pos):
                            index = i
                            break
                    # Remove the face from the list
                    if index >= 0:
                        faces.pop(index)
                        bboxes = np.delete(bboxes, index, axis=0)
                        kpss = np.delete(kpss, index, axis=0)
                    else:
                        print(f"WARNING: Could not find the face of person {person.attrib['id']} in frame {frame_n} of camera {cam_n}")
                # If there are faces left, save them as unknown faces
                for i, face in enumerate(faces):
                    left_eye_pos = kpss[i][0]
                    right_eye_pos = kpss[i][1]
                    # Draw bbox around the face
                    bbox = [int(x) for x in bboxes[i]]
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                    # Save the face image
                    cv2.imwrite(os.path.join(dest_path, f"C{camera+1}", f"unknown_face_{frame_n}.jpg"), img)
                    # Create a tag <person id="Unknown">
                    person = ET.Element("person")
                    person.attrib["id"] = "Unknown"
                    # Create a tag <leftEye x="x" y="y"/>
                    left_eye = ET.SubElement(person, "leftEye")
                    left_eye.attrib["x"] = str(int(left_eye_pos[0]))
                    left_eye.attrib["y"] = str(int(right_eye_pos[1]))
                    # Create a tag <rightEye x="x" y="y"/>
                    right_eye = ET.SubElement(person, "rightEye")
                    right_eye.attrib["x"] = str(int(right_eye_pos[0]))
                    right_eye.attrib["y"] = str(int(right_eye_pos[1]))
                    # Add the person as child of frame element
                    frame.append(person)
            # Save the groundtruth
            tree.write(groundtruth_file)

if __name__ == '__main__':
    main()