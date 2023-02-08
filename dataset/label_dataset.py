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
    # Clear the temp_unknown_faces folder
    dest_path = os.path.join("temp_unknown_faces", TEST_SCENARIO, TEST_SCENARIO2)
    shutil.rmtree(dest_path, ignore_errors=True)
    os.makedirs(dest_path, exist_ok=True)

    # For each camera find the unknown faces
    for camera in range(MAX_CAMERAS):
        # Make dirs
        os.makedirs(os.path.join(dest_path, f"C{camera+1}"), exist_ok=True)
        path = os.path.join(TEST_PATH, f"{TEST_SCENARIO}_C{camera+1}{TEST_SCENARIO2}") # path of the camera

        # Load the groundtruth
        groundtruth_file = os.path.join(GROUNDTRUTH_PATH, f"{TEST_SCENARIO}_C{camera+1}{TEST_SCENARIO2}.xml")
        tree = ET.parse(groundtruth_file)
        
        # For each frame in the groundtruth, autodetect images and check if there are unknown faces
        print(f"Detecting unknown faces in {TEST_SCENARIO}_C{camera+1}{TEST_SCENARIO2}...")
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
                # # Get name of the person
                # name = person.attrib["id"]
                # # If the name is Unknown, skip
                # if name == "Unknown":
                #     continue
                # Get the left eye position
                left_eye = person.find("leftEye")
                left_eye_pos = (int(left_eye.attrib["x"]), int(left_eye.attrib["y"]))
                # Get the right eye position
                right_eye = person.find("rightEye")
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
            # If the face is not similar, save it as an Unknown face
            if len(faces) > 0:
                left_eye_pos = kpss[0][0]
                right_eye_pos = kpss[0][1]
                # Draw bbox around the face
                bbox = [int(x) for x in bboxes[0]]
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                # Save the face image
                cv2.imwrite(os.path.join(dest_path, f"C{camera+1}", f"unknown_face_{frame_n}.jpg"), img)
                # Create a tag <person id="Unknown">
                person = ET.Element("person")
                person.attrib["id"] = "Unknown"
                # Create a tag <leftEye x="x" y="y"/>
                left_eye = ET.SubElement(person, "leftEye")
                left_eye.attrib["x"] = str(left_eye_pos[0])
                left_eye.attrib["y"] = str(right_eye_pos[1])
                # Create a tag <rightEye x="x" y="y"/>
                right_eye = ET.SubElement(person, "rightEye")
                right_eye.attrib["x"] = str(right_eye_pos[0])
                right_eye.attrib["y"] = str(right_eye_pos[1])
                # Add the person as child of frame element
                frame.append(person)
        # Save the groundtruth
        tree.write(groundtruth_file)

if __name__ == '__main__':
    main()