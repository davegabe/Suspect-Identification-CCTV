import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from insight_utilities.insight_interface import get_faces

from modules.gallery_module import Identity, build_gallery, check_identity, Identity
from dataset.dataset import protocols

def draw(identities, bboxes, kpss, camera_img, frame, num_cam):
    for i, identity in enumerate(identities):
        print("In frame " + frame + " the identity is " + identity)
        # Draw the bouding box in plt
        x1 = int(bboxes[i][0])
        y1 = int(bboxes[i][1])
        x2 = int(bboxes[i][2])
        y2 = int(bboxes[i][3])
        # Draw the keypoints
        for kp in kpss[i]:
            cv2.circle(camera_img, (int(kp[0]), int(kp[1])), 1, (0, 0, 255), 1)

        cv2.rectangle(camera_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Print the identity reducing the size of the text to be minor than AA
        cv2.putText(camera_img, identity, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # Save the image
        plt.imshow(camera_img)
        plt.title("Identity: " + identity)
        plt.imsave('results/' + f"_{num_cam}_" + frame + ".png", camera_img)
        plt.close()

def handle_frame(camera_images: list[np.ndarray], gallery: dict, frame: str):
    unknown_identities: list[Identity] = []
    known_identities: list[Identity] = []
    # For each camera 
    for num_cam, camera_img in enumerate(camera_images):
        # Extract faces from the frame
        faces, bboxes, kpss = get_faces(camera_img)
        # For each face in the frame, get the unknown identity
        for face, bbox, kps in zip(faces, bboxes, kpss):
            # Compare the frame with the gallery
            # identities, bboxes, kpss = check_identity(gallery, camera_img)
            pass

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
        print(f"Current environment: {environment} using {str(max_cameras)} cameras")
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
        detectedIdentities = dict()
        """"
        Per ogni frame e per ogni telecamera:
            faccio reIdentification
            Così mi trovo in primis il suo ID temporanei
            Lavorerò con dei ID temporanei fino alla fine della esecuzione. 
        Una volta finito, cercherò dei ID che possono essere match e poi salverò i risultati

        Problema di cui tener conto: se ci sono piu facce unknown, rischiano di confondersi
        """
        # For each frame
        for frame in frames:
            camera_images: list[np.ndarray] = [cv2.imread(os.path.join(path, frame)) for path in paths]
            handle_frame(camera_images, gallery, frame)

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