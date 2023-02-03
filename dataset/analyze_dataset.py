import os
import cv2
import matplotlib.pyplot as plt

from config import TEST_PATH, TEST_SCENARIO
from insight_utilities.insight_interface import get_face

def main():
    # Get the list of frames
    frames = sorted(filter(lambda x: x.endswith(".jpg"), os.listdir(os.path.join(TEST_PATH, TEST_SCENARIO + "_C1"))))[760::8]
    # Autodetect faces in the frames
    img_faces = []
    for frame in frames:
        print(os.path.join(TEST_PATH, TEST_SCENARIO + "_C1", frame))
        # Get the frame
        frame = cv2.imread(os.path.join(TEST_PATH, TEST_SCENARIO + "_C1", frame))
        # Swap the BGR frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get the faces
        print(frame.shape)
        face, bbox, kps = get_face(frame)
        if face is None:
            continue
        # Crop the face
        x1 = int(bbox[0][0])
        y1 = int(bbox[0][1])
        x2 = int(bbox[0][2])
        y2 = int(bbox[0][3])
        face = frame[y1:y2, x1:x2]
        img_faces.append(face)
        # If img_faces has more than 6 elements, stop
        if len(img_faces) >= 6:
            break
    # Save the images
    for i, img in enumerate(img_faces):
        plt.imsave(f"dataset/plots/face_{i}.png", img)
        
if __name__ == "__main__":
    main()