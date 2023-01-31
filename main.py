import os
import cv2
import matplotlib.pyplot as plt

from modules.gallery_module import build_gallery, check_identity
from modules.frame_selection_module import select_best_frame
from dataset.dataset import protocols



def main():
    dataset_path = "data"
    for environment in  protocols.keys():
        # Get an arbitrary scenario
        scenario = protocols[environment][0]
        # Get the max_cameras cameras
        max_cameras = 3
        cameras = ["C" + str(i) for i in range(1, max_cameras)]
        # Get the paths for each camera
        paths = [os.path.join("data", environment, f"{environment}_{scenario}_{camera}") for camera in cameras]
        print("Current environment: " + environment + " using " + str(max_cameras) + " cameras")
        # Get the other environments
        other_environments = list(filter(lambda x: x != environment, protocols.keys()))
        # Get an arbitrary environment for the gallery
        environment_for_gallery = other_environments[0]
        # Get an arbitrary scenario for the gallery
        scenario_for_gallery = protocols[environment_for_gallery][0]
        # Build the gallery
        gallery_path = os.path.join(dataset_path, environment_for_gallery)
        gallery = build_gallery(gallery_path, f"{environment_for_gallery}_{scenario_for_gallery}_C1", max_size=15)
        # Load all frames
        frames = list(filter(lambda x: x.endswith(".jpg"), os.listdir(paths[0])))
        frames = sorted(frames)
        # For each frame
        for frame in frames:
            camera_frames = [cv2.imread(os.path.join(path, frame)) for path in paths]
            # Pick the best frame
            best_frame = select_best_frame(camera_frames)
            # Compare the frame with the gallery
            identities, bboxes = check_identity(gallery, best_frame)
            for i, identity in enumerate(identities):
                print("In frame " + frame + " the identity is " + identity)
                # Draw the bouding box in plt
                x1 = int(bboxes[i][0])
                y1 = int(bboxes[i][1])
                x2 = int(bboxes[i][2])
                y2 = int(bboxes[i][3])
                cv2.rectangle(best_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # Print the identity reducing the size of the text to be minor than AA
                cv2.putText(best_frame, identity, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # Save the image
                plt.imshow(best_frame)
                plt.title("Identity: " + identity)
                plt.imsave('results/' + frame + ".png", best_frame)
                plt.close()
        # TODO: remove the following line to test all the environments
        break
        



if __name__ == "__main__":
    # img = cv2.imread("data/P1E/P1E_S1_C1/00000252.jpg")
    # from insight_utilities.scrfd import SCRFD 
    # import os.path as osp
    # assets_dir = osp.expanduser('~/.insightface/models/buffalo_l')
    # detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
    # detector.prepare(0)
    # print(detector.autodetect(img, max_num=2))
    main()
