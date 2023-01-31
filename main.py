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
        #lista di sospetti
        # For each frame
        for frame in frames:
            camera_frames = [cv2.imread(os.path.join(path, frame)) for path in paths]
            # Pick the best frame
            best_frame = select_best_frame(camera_frames)
            # Compare the frame with the gallery
            identities, bboxes, kpss = check_identity(gallery, best_frame)
            for i, identity in enumerate(identities):
                print("In frame " + frame + " the identity is " + identity)
                # Draw the bouding box in plt
                x1 = int(bboxes[i][0])
                y1 = int(bboxes[i][1])
                x2 = int(bboxes[i][2])
                y2 = int(bboxes[i][3])
                # Draw the keypoints
                for kp in kpss[i]:
                    cv2.circle(best_frame, (int(kp[0]), int(kp[1])), 1, (0, 0, 255), 1)

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



"""
Tracciare continuamente una persona con una label temporanea, fregandoci di chi è effetivamente.  L'intuizione è che una persona verosimilmente non si
può teletrasportare quindi i frame successivi a quello in cui è stata identificata, avranno una similarità quasi identica a quello precedente.
Per alleggerire il calcolo, si può usare una funzione di similarità che sfrutta solo gli ultimi X frames
Quando la persona scompare dalla scena, la label temporanea viene sostituita con una label definitiva (che viene assegnata a tutti i frame precedenti).
"""

"""

temp_identities = {
    "identity_temp": {
        "frames": [frame1, frame2, frame3,  ...],     # each element of this (and also the following) list is a list of results for each camera (list of 3 elements)
        "bboxes": [bbox1, bbox2, bbox3, ...],
        "kpss": [kps1, kps2, kps3, ...],
        "features": [feature1, feature2, feature3, ...]
        "is_in_scene": True  # this will be False when the person disappears from the scene, at this point the decision module will do stuff and replace the temporary identity with a definitive one
    }
}

for frame in frames:
    for camera in cameras:
        # for each face, i check if this face is in temp_identities
            # if it is, i update the temp_identity (add the frame, the bboxes, the kpss, the features)
            # if it is not, i create a new temp_identity
    # at the end of the loop, if there is a face in temp_identities (maybe we have to wait for more than 1 frame where the face is not present) that is not in the current frame, i set is_in_scene to False
    # decision module will check if there are not is_in_scene temp_identities and will decide what their permanent identity is (removing them from temp_identities and putting somewhere else)
    # printing the results somehow



for key in temp_identities.keys():
    for frame in dict[key]["frames"]:
        #detecta faccia
        #vedi quale è la più simile
"""