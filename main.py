from multiprocessing import Queue
import os
from pprint import pprint
import cv2
import numpy as np
from insight_utilities.insight_interface import get_faces

from modules.drawing_module import GUI
from modules.evaluation_module import build_groundtruth, evaluate_system
from modules.gallery_module import Identity, build_gallery, Identity
from modules.decision_module import decide_identities
from config import TEST_PATH, TEST_SCENARIO, TEST_SCENARIO2, UNKNOWN_SIMILARITY_THRESHOLD, MAX_CAMERAS

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
    # for y in os.listdir("data/"):
    #     if y.endswith("_faces"):
    #         for x in os.listdir(f"data/{y}"):
    #             print(f"data/{y}/{x}\n")
    #             a = build_groundtruth(f"data/{y}/{x}")
    #             for i, j in a.items():
    #                 if len(j) > 1:
    #                     pprint(f"{i}: {j}")
    #                     break

    # for x in os.listdir("data/groundtruth"):
    #     a = build_groundtruth(f"data/groundtruth/{x}")
    #     for i, j in a.items():
    #         if len(j) > 1:
    #             pprint(f"{x} --> {i}: {j}")
    #             break
    
    # Build the gallery
    pprint(build_groundtruth(["data/groundtruth/P1E_S1_C1.xml", "data/groundtruth/P1E_S1_C2.xml", "data/groundtruth/P1E_S1_C3.xml"]))
    print("Building the gallery...")
    gallery = build_gallery()

    # Initialize the identities
    unknown_identities: list[Identity] = [] # temporary identities which don't have a label yet
    known_identities: list[Identity] = [] # permanent identities which have a label

    # Load all frames
    print("Loading frames...")
    paths = [os.path.join(TEST_PATH, f"{TEST_SCENARIO}_C{i+1}{TEST_SCENARIO2}") for i in range(MAX_CAMERAS)] # paths of the cameras
    frames = list(filter(lambda x: x.endswith(".jpg"), os.listdir(paths[0]))) # frames of the cameras
    frames = sorted(frames)
    frames_reduced = frames[130:int(len(frames)*0.2)] # frames to be processed
    all_camera_images = [[cv2.imread(os.path.join(path, frame)) for path in paths] for frame in frames_reduced] # images of the cameras

    # Initialize gui queues
    requests_queue = Queue()
    responses_queue = Queue()

    # Launch the GUI
    print("Launching GUI...")
    # Launch the GUI on a separate thread
    guip = GUI(requests_queue, responses_queue, len(frames_reduced))
    guip.start()

    for i, frame_name in enumerate(frames_reduced):
        print(f"Current frame: {frame_name}")
        handle_frame(all_camera_images[i], gallery, unknown_identities, known_identities, i)
        handle_gui_communication(all_camera_images, unknown_identities, known_identities, requests_queue, responses_queue, i)

    # Force last decision
    unknown_identities, known_identities = decide_identities(unknown_identities, known_identities, gallery, force=True)

    # # Draw result images
    # draw_files(known_identities + unknown_identities, frames_reduced, paths)

    # Evaluate the results
    evaluate_system(known_identities, unknown_identities, [os.path.join("data/groundtruth/", f"{TEST_SCENARIO}_C{i+1}{TEST_SCENARIO2}.xml") for i in range(MAX_CAMERAS)], frames_reduced)
    
    # Wait for the GUI to close while communicating with it
    while True:
        handle_gui_communication(all_camera_images, unknown_identities, known_identities, requests_queue, responses_queue, len(frames_reduced))
        if not guip.is_alive():
            break
        

if __name__ == "__main__":
    main()
