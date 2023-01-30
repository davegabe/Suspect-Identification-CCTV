import os
import numpy as np
import cv2
import sys

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
        frame_img = cv2.imread(os.path.join(path, frame), cv2.IMREAD_COLOR)
        #the channel is BGR, convert to RGB
        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        # Compare the frame with the gallery
        identity, bboxes = check_identity(gallery, frame_img)
        if identity is not None:
            print("In frame " + frame + " the identity is " + identity)
            #draw the bounding box
            

            #draw the bouding box in plt
            for bbox in bboxes:
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])

                #draw the bounding box
                cv2.rectangle(frame_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                #print the identity reducing the size of the text to be minor than AA
                cv2.putText(frame_img, identity, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)



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
