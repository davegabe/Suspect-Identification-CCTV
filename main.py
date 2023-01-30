import os
import numpy as np
import cv2

from modules.gallery_module import build_gallery, check_identity

#import things for plotting and image
import matplotlib.pyplot as plt


def main():
    dataset_path = "data"
    environments = list(filter(lambda x: os.path.isdir(os.path.join(dataset_path, x)) and not x.endswith("_faces"), os.listdir(dataset_path)))
    # for environment in environments:
    environment = environments[0]
    camera = os.listdir("data/" + environment)[0]
    path = os.path.join("data", environment, camera)
    other_environments = environments.copy()
    other_environments.remove(environment)
    print("Current environment: " + environment + " and camera: " + camera + " and path: " + path)
    # Create a gallery
    environment_for_gallery = other_environments[np.random.randint(0, len(other_environments))]
    gallery = build_gallery(os.path.join(dataset_path, environment_for_gallery), max_size=10)
    # Load all frames
    frames = sorted(os.listdir(path))
    # For each frame
    for frame in frames:
        # Get the frame
        frame_img = cv2.imread(os.path.join(path, frame))
        # Compare the frame with the gallery
        identity, boundingBoxes, keyPoints = check_identity(gallery, frame_img)
        if identity is not None:
            print("In frame " + frame + " the identity is " + identity)

            #draw bounding box
            print(boundingBoxes)
            for box in boundingBoxes:
                cv2.rectangle(frame_img, (box[0][0], box[0][1]), (box[0][2], box[0][3]), (0, 255, 0), 2)
            #draw keypoints
            for keypoint in keyPoints:
                cv2.circle(frame_img, (keypoint[0], keypoint[1]), 2, (0, 0, 255), 2)
            plt.imshow(frame_img)
            plt.title("Identity: " + identity)
            plt.imsave('results/' + frame + ".png", frame_img)
            plt.close()
        



if __name__ == "__main__":
    # img = cv2.imread("data/P1E/P1E_S1_C1/00000252.jpg")
    # from insight_utilities.scrfd import SCRFD 
    # import os.path as osp
    # assets_dir = osp.expanduser('~/.insightface/models/buffalo_l')
    # detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
    # detector.prepare(0)
    # print(detector.autodetect(img, max_num=2))
    main()
