import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import shutil

from insight_utilities.insight_interface import get_face, get_faces, compareTwoFaces
from config import TEST_PATH, TEST_SCENARIO, TEST_SCENARIO2, MAX_CAMERAS, GALLERY_THRESHOLD

class UnknownFace:
    """
    Unknown face class
    """
    def __init__(self, camera: str, frame: str, left_eye: tuple[int, int], right_eye: tuple[int, int]):
        self.frame = frame
        self.camera = camera
        self.left_eye = left_eye
        self.right_eye = right_eye

def add_in_groundtruth(groundtruth_path: list[str], unknown_faces: list[UnknownFace]):
    """
    Add the unknown faces to the groundtruth file.
    """
    cam_n = os.path.basename(groundtruth_path).split("_")[-1].split(".")[0][1] # Assumes the name of the file is "ENV_SN_CN.n.xml"
    tree = ET.parse(groundtruth_path)
    for frame in tree.getroot():
        frame_n = frame.attrib["number"]
        # Check if there is an unknown face in this frame
        for unknown_face in unknown_faces:
            if int(unknown_face.camera) == int(cam_n) and int(unknown_face.frame) == int(frame_n):
                print(f"Adding unknown face to frame {frame_n} of camera {cam_n}")
                # Create a tag <person id="Unknown">
                person = ET.Element("person")
                person.attrib["id"] = "Unknown"
                # Create a tag <leftEye x="x" y="y"/>
                left_eye = ET.SubElement(person, "leftEye")
                left_eye.attrib["x"] = str(unknown_face.left_eye[0])
                left_eye.attrib["y"] = str(unknown_face.left_eye[1])
                # Create a tag <rightEye x="x" y="y"/>
                right_eye = ET.SubElement(person, "rightEye")
                right_eye.attrib["x"] = str(unknown_face.right_eye[0])
                right_eye.attrib["y"] = str(unknown_face.right_eye[1])
                # Add the person as child of frame element
                frame.append(person)
    # Save the new groundtruth
    tree.write(groundtruth_path)

def main():
    # Clear the temp_unknown_faces folder
    dest_path = os.path.join("temp_unknown_faces", TEST_SCENARIO, TEST_SCENARIO2)
    shutil.rmtree(dest_path, ignore_errors=True)
    os.makedirs(dest_path, exist_ok=True)

    # For each camera find the unknown faces
    for camera in range(MAX_CAMERAS):
        # Make dirs
        os.makedirs(os.path.join(dest_path, f"C{camera+1}"), exist_ok=True)
        # Load all frames
        print("Loading frames...")
        path = os.path.join(TEST_PATH, f"{TEST_SCENARIO}_C{camera+1}{TEST_SCENARIO2}") # paths of the cameras
        frames = list(filter(lambda x: x.endswith(".jpg"), os.listdir(path))) # frames of the cameras
        frames = sorted(frames)[:500]
        all_camera_images = [cv2.imread(os.path.join(path, frame)) for frame in frames] # images of the cameras
        frames = [frame_name.replace(".jpg", "") for frame_name in frames]

        # Load the face images
        print("Loading face images...")
        faces_path = os.path.join(f"{TEST_PATH}_faces", f"{TEST_SCENARIO}_C{camera+1}{TEST_SCENARIO2}") # path of the faces
        faces_by_frame: dict[str, list[tuple[str, np.ndarray]]] = dict() # dict of the faces for each frame
        identities = os.listdir(faces_path) # identities of the faces
        # For each frame append all the (name, face_features) tuples to the list
        for frame_name in frames:
            frame_faces = []
            for identity in identities:
                # Get the pgm file and append it to the list
                if not os.path.exists(os.path.join(faces_path, identity, f"{frame_name}.pgm")):
                    continue
                face_path = os.path.join(faces_path, identity, f"{frame_name}.pgm")
                face, _, _ = get_face(cv2.imread(face_path))
                if face is None:
                    print(f"Face not found in {face_path}, this should not happen")
                    continue
                frame_faces.append((identity, face))
            faces_by_frame[frame_name] = frame_faces
        
        # For each frame in the camera, autodetect images and check if there are unknown faces
        print("Autodetecting faces...")
        unknown_faces: list[UnknownFace] = []
        for frame_name, img in zip(frames, all_camera_images):
            # Get the faces from the frame
            faces, bboxes, kpss = get_faces(img)
            if len(faces) == 0:
                continue
            # For each face, check if it similar to any of the known faces
            for face, bbox, kps in zip(faces, bboxes, kpss):
                # Check if the face is similar to any of the known faces
                similar = False
                for known_face in faces_by_frame[frame_name]:
                    # Check if the face is similar to the known face
                    if compareTwoFaces(face, known_face[1]) > GALLERY_THRESHOLD:
                        similar = True
                        break
                # If the face is not similar, save it as an unknown face
                if not similar:
                    left_eye_pos = kps[0]
                    right_eye_pos = kps[1]
                    # print(f"Unknown face detected in frame {frame_name} of camera {camera+1}, eyes at {left_eye_pos} and {right_eye_pos}")
                    # Draw bbox around the face
                    bbox = [int(x) for x in bbox]
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                    # Save the face image
                    cv2.imwrite(os.path.join(dest_path, f"C{camera+1}", f"unknown_face_{frame_name}.jpg"), img)
                    # Save the unknown face
                    unknown_faces.append(UnknownFace(camera+1, frame_name, left_eye_pos, right_eye_pos))
        
        # Add the unknown faces to the groundtruth
        print("Adding unknown faces to the groundtruth...")
        groundtruth_path = os.path.join(f"a_C{camera+1}.xml")
        add_in_groundtruth(groundtruth_path, unknown_faces)

if __name__ == '__main__':
    main()