import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from insight_utilities.insight_interface import get_faces

from modules.evaluation_module import build_groundtruth, evaluate_system
from modules.gallery_module import Identity, build_gallery, Identity
from modules.decision_module import decide_identities
from dataset.dataset import protocols
from config import UNKNOWN_SIMILARITY_THRESHOLD, MAX_CAMERAS

"""
for each frame
        for each identity  
            see if frame is inside identity
                if yes
                    draw bbox and name
                if no
                    continue
"""

def draw(identities: list[Identity], frames: list[str], paths: list[str]):
    # For each frame
    for frame in frames:
        camera_images: list[np.ndarray] = [cv2.imread(os.path.join(path, frame)) for path in paths]
        # For each camera image
        for num_cam, camera_img in enumerate(camera_images):
            found = False
            #Find the identity in the frame, draw the bbox and the name
            for i, identity in enumerate(identities):
                if f"{num_cam}_{frame}" in identity.frames:
                    i = identity.frames.index(f"{num_cam}_{frame}")
                    print(f"In frame {frame} of camera {num_cam}, the identity is {identity.name}")
                    # Draw the bouding box in plt
                    x1 = int(identity.bboxes[i][0])
                    y1 = int(identity.bboxes[i][1])
                    x2 = int(identity.bboxes[i][2])
                    y2 = int(identity.bboxes[i][3])
                    # Draw the keypoints
                    for kp in identity.kps[i]:
                        cv2.circle(camera_img, (int(kp[0]), int(kp[1])), 1, (0, 0, 255), 1)

                    cv2.rectangle(camera_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # Print the identity reducing the size of the text to be minor than AA
                    cv2.putText(camera_img, identity.name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    found = True
            # Save the image
            if found:
                plt.imshow(camera_img)
                plt.imsave('results/' + f"_{num_cam}_" + frame + ".png", camera_img)
                plt.close()


def handle_frame(camera_images: list[np.ndarray], gallery: dict, unknown_identities: list[Identity], known_identities: list[Identity], frame: str):
    found_identities: list[Identity] = [] # identities which have been found in the current frame
    # For each camera image
    for num_cam, camera_img in enumerate(camera_images):
        # Extract faces from the image
        faces, bboxes, kpss = get_faces(camera_img)
        # For each face in the frame, get the unknown identity
        for face, bbox, kps in zip(faces, bboxes, kpss):
            # We have to check if the face is similar to an unknown identity
            similarity = [identity.match(face) for identity in unknown_identities]
            max_similarity = max(similarity + [0])
            if max_similarity > UNKNOWN_SIMILARITY_THRESHOLD:
                # Get the index of the identity with the max similarity
                index = similarity.index(max_similarity)
                # Get the identity with the max similarity
                identity = unknown_identities[index]
                # Add the frame to the identity
                identity.add_frame(face, bbox, kps, f"{num_cam}_{frame}")
                # Add the identity to the list of found identities
                found_identities.append(identity)
            else:
                # Create a new identity
                new_identity = Identity()
                new_identity.add_frame(face, bbox, kps, f"{num_cam}_{frame}")
                unknown_identities.append(new_identity)
    # For each unknown identity, check if it has been found in the current frame
    for unknown_identity in unknown_identities:
        # If the identity has been found, add it to the known identities
        if unknown_identity not in found_identities:
            unknown_identity.max_missing_frames -= 1
    # Decision module
    unknown_identities, known_identities = decide_identities(unknown_identities, known_identities, gallery)

def main():
    dataset_path = "data"
    for environment in  protocols.keys():
        # Get an arbitrary scenario
        scenario = protocols[environment][0]
        # Get the max_cameras cameras
        cameras = ["C" + str(i) for i in range(1, MAX_CAMERAS+1)]
        # Get the paths for each camera
        paths = [os.path.join("data", environment, f"{environment}_{scenario}_{camera}") for camera in cameras]
        print(f"Current environment: {environment} on scenario {scenario} using {str(MAX_CAMERAS)} cameras")
        # Get the other environments
        other_environments = list(filter(lambda x: x != environment, protocols.keys()))
        # Get an arbitrary environment for the gallery
        environment_for_gallery = other_environments[0]
        # Get an arbitrary scenario for the gallery
        scenario_for_gallery = protocols[environment_for_gallery][0]

        # Build the gallery
        print("Building the gallery...")
        gallery_path = os.path.join(dataset_path, environment_for_gallery)
        gallery = build_gallery(gallery_path, f"{environment_for_gallery}_{scenario_for_gallery}_C1")

        # Load all frames
        print("Loading frames...")
        frames = list(filter(lambda x: x.endswith(".jpg"), os.listdir(paths[0])))
        frames = sorted(frames)

        # For each frame
        unknown_identities: list[Identity] = [] # temporary identities which don't have a label yet
        known_identities: list[Identity] = [] # permanent identities which have a label
        print(paths)
        frames_reduced = frames[130:int(len(frames)*0.2)]
        for frame in frames_reduced:
            print(f"Current frame: {frame}")
            camera_images: list[np.ndarray] = [cv2.imread(os.path.join(path, frame)) for path in paths]
            handle_frame(camera_images, gallery, unknown_identities, known_identities, frame)
        # Force last decision
        unknown_identities, known_identities = decide_identities(unknown_identities, known_identities, gallery, force=True)
        print(len(known_identities), len(frames_reduced))
        # Draw result images
        draw(known_identities + unknown_identities, frames_reduced, paths)
        # Evaluate the results
        print(f"Evaluation for {environment} using {str(MAX_CAMERAS)} cameras")
        evaluate_system(known_identities, os.path.join(dataset_path, f"{environment}_faces"))
        # TODO: remove the following line to test all the environments
        break
        

if __name__ == "__main__":
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
        "frames": [[frame1, frame2, frame3], [frame2...], [frame3...],  ...],     # each element of this (and also the following) list is a list of results for each camera (list of 3 elements)
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