from multiprocessing import Queue
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from insight_utilities.insight_interface import get_faces

from modules.drawing_module import GUI
from modules.evaluation_module import evaluate_system
from modules.gallery_module import Identity, build_gallery, Identity
from modules.decision_module import decide_identities
from dataset.dataset import protocols
from config import UNKNOWN_SIMILARITY_THRESHOLD, MAX_CAMERAS

def draw(identities: list[Identity], frames: list[str], paths: list[str]):
    """
    This function draws the identities in the frames with the name and the bbox.

    Args:
        identities (list[Identity]): The identities
        frames (list[str]): The frames
        paths (list[str]): The paths of the cameras
    """
    # For each frame
    for frame in frames:
        camera_images: list[np.ndarray] = [cv2.imread(os.path.join(path, frame)) for path in paths]
        # For each camera image
        for num_cam, camera_img in enumerate(camera_images):
            found = False
            # Find the identity in the frame, draw the bbox and the name
            for i, identity in enumerate(identities):
                # Check if the frame is in the identity and draw the bbox and the name
                for i in range(len(identity.frames)):
                    if identity.frames[i] == f"{num_cam}_{frame}":
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
                        cv2.putText(camera_img, f"{identity.name}, id:{identity.id}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                found = True
            # Save the image
            if found:
                plt.imshow(camera_img)
                plt.imsave('results/' + f"_{num_cam}_" + frame + ".png", camera_img)
                plt.close()

def handle_gui_communication(all_camera_images: list[list[np.ndarray]], unknown_identities: list[Identity], known_identities: list[Identity], requests_queue: Queue, responses_queue: Queue, curr_frame: int):
    """
    This function handles the communication between the GUI and the main thread.

    Args:
        all_camera_images (list[list[np.ndarray]]): The images of the cameras
        unknown_identities (list[Identity]): The unknown identities
        known_identities (list[Identity]): The known identities
        requests_queue (Queue): The queue of requests
        responses_queue (Queue): The queue of responses
    """
    # If there is a request
    if not requests_queue.empty():
        # Get the request
        frame, camera = requests_queue.get()
        # If requested frame is not processed yet
        if frame > curr_frame:
            # Create a black image
            camera_img = np.zeros(all_camera_images[0][0].shape, dtype=np.uint8)
            # Write the text in center
            cv2.putText(camera_img, "Frame not available", (int(camera_img.shape[1]/2), int(camera_img.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # Get the identities in the frame
            responses_queue.put((camera_img, [], []))
        else:
            # Get the image of the camera
            camera_img = all_camera_images[frame][camera]
            # Get the identities in the frame
            responses_queue.put((camera_img, known_identities, unknown_identities))


def handle_frame(camera_images: list[np.ndarray], gallery: dict, unknown_identities: list[Identity], known_identities: list[Identity], frame: int):
    """
    This function handles a frame, extracting the faces, matching them with the identities and updating the identities.

    Args:
        camera_images (list[np.ndarray]): The images of the cameras
        gallery (dict): The gallery
        unknown_identities (list[Identity]): The unknown identities
        known_identities (list[Identity]): The known identities
    """
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
        gallery_scenario_camera = f"{environment_for_gallery}_{scenario_for_gallery}_C1"
        gallery = build_gallery(gallery_path, gallery_scenario_camera)

        # Initialize the identities
        unknown_identities: list[Identity] = [] # temporary identities which don't have a label yet
        known_identities: list[Identity] = [] # permanent identities which have a label

        # Load all frames
        print("Loading frames...")
        frames = list(filter(lambda x: x.endswith(".jpg"), os.listdir(paths[0])))
        frames = sorted(frames)
        frames_reduced = frames[130:int(len(frames)*0.2)]
        all_camera_images = [[cv2.imread(os.path.join(path, frame)) for path in paths] for frame in frames_reduced]

        # Initialize gui queues
        requests_queue = Queue()
        responses_queue = Queue()

        # Launch the GUI
        print("Launching GUI...")
        # Launch the GUI on a separate thread
        guip = GUI(requests_queue, responses_queue, len(frames_reduced), gallery_path)
        guip.start()

        for i, frame_name in enumerate(frames_reduced):
            print(f"Current frame: {frame_name}")
            handle_frame(all_camera_images[i], gallery, unknown_identities, known_identities, i)
            handle_gui_communication(all_camera_images, unknown_identities, known_identities, requests_queue, responses_queue, i)
        # Force last decision
        unknown_identities, known_identities = decide_identities(unknown_identities, known_identities, gallery, force=True)
        # # Draw result images
        # draw(known_identities + unknown_identities, frames_reduced, paths)
        # Evaluate the results
        print(f"Evaluation for {environment} using {str(MAX_CAMERAS)} cameras")
        # evaluate_system(known_identities, unknown_identities, os.path.join(dataset_path, f"{environment}_faces"))
        # Wait for the GUI to close while communicating with it
        while True:
            handle_gui_communication(all_camera_images, unknown_identities, known_identities, requests_queue, responses_queue, len(frames_reduced))
            if not guip.is_alive():
                break
        # TODO: remove the following line to test all the environments
        break
        

if __name__ == "__main__":
    main()
