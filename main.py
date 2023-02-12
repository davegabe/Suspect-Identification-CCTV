import json
from multiprocessing import Queue
from time import time
import os
import cv2
import numpy as np
import gc
from tqdm import tqdm

from modules.drawing_module import GUI
from modules.evaluation_module import evaluate_system
from modules.gallery_module import Identity, build_gallery, Identity
from modules.decision_module import decide_identities
from insight_utilities.insight_interface import get_faces
from config import TEST_PATH, TEST_SCENARIO, TEST_SCENARIO2, UNKNOWN_SIMILARITY_THRESHOLD, MAX_CAMERAS, SEED, USE_GUI, GALLERY_THRESHOLD

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


def handle_frame(camera_images: list[np.ndarray], gallery: dict, unknown_identities: list[Identity], known_identities: list[Identity], frame_name: str, gallery_threshold = GALLERY_THRESHOLD):
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
                identity.add_frame(face, bbox, kps, f"{num_cam+1}_{frame_name}")
                # Add the identity to the list of found identities
                found_identities.append(identity)
            else:
                # Create a new identity
                new_identity = Identity()
                new_identity.add_frame(face, bbox, kps, f"{num_cam+1}_{frame_name}")
                unknown_identities.append(new_identity)
    # For each unknown identity, check if it has been found in the current frame
    for unknown_identity in unknown_identities:
        # If the identity has been found, add it to the known identities
        if unknown_identity not in found_identities:
            unknown_identity.max_missing_frames -= 1
    # Decision module
    unknown_identities, known_identities = decide_identities(unknown_identities, known_identities, gallery, gallery_threshold)

def main():
    np.random.seed(SEED) # ඞ

    # Build the gallery
    print("Building the gallery...")
    groundtruth_paths = [os.path.join("data/groundtruth/", f"{TEST_SCENARIO}_C{i+1}{TEST_SCENARIO2}.xml") for i in range(MAX_CAMERAS)]
    gallery, gallery_sample = build_gallery(groundtruth_paths)

    # Initialize the identities
    unknown_identities: list[Identity] = [] # temporary identities which don't have a label yet
    known_identities: list[Identity] = [] # permanent identities which have a label

    # Load all frames
    print("Loading frames...")
    paths = [os.path.join(TEST_PATH, f"{TEST_SCENARIO}_C{i+1}{TEST_SCENARIO2}") for i in range(MAX_CAMERAS)] # paths of the cameras
    frames = list(filter(lambda x: x.endswith(".jpg"), os.listdir(paths[0]))) # frames of the cameras
    frames = sorted(frames)
    frames_reduced = frames[136:int(len(frames)*0.2)] # frames to be processed
    all_camera_images = [[cv2.imread(os.path.join(path, frame)) for path in paths] for frame in frames_reduced] # images of the cameras

    # Initialize gui queues
    requests_queue = Queue()
    responses_queue = Queue()

    # Launch the GUI
    print("Launching GUI...")
    # Launch the GUI on a separate thread
    all_frames_no_cameras = list(map(lambda x: x.split(".")[0], frames_reduced))
    if USE_GUI:
        guip = GUI(requests_queue, responses_queue, len(frames_reduced), all_frames_no_cameras, gallery_sample)
        guip.start()

    i = 0
    for frame_name in tqdm(all_frames_no_cameras, desc="Processing frames"):
        handle_frame(all_camera_images[i], gallery, unknown_identities, known_identities, frame_name)
        if USE_GUI:
            handle_gui_communication(all_camera_images, unknown_identities, known_identities, requests_queue, responses_queue, i)
        i += 1

    # Force last decision
    unknown_identities, known_identities = decide_identities(unknown_identities, known_identities, gallery, GALLERY_THRESHOLD, force=True)

    # Evaluate the results
    all_frames_cameras = []
    for i in range(MAX_CAMERAS):
        for frame in map(lambda x: x.split(".")[0], frames_reduced):
            all_frames_cameras.append(f"{i+1}_{frame}")
    groundtruth_paths = [os.path.join("data/groundtruth/", f"{TEST_SCENARIO}_C{i+1}{TEST_SCENARIO2}.xml") for i in range(MAX_CAMERAS)]
    evaluate_system(known_identities, unknown_identities, groundtruth_paths, all_frames_cameras, gallery)
    
    # Wait for the GUI to close while communicating with it
    while True and USE_GUI:
        handle_gui_communication(all_camera_images, unknown_identities, known_identities, requests_queue, responses_queue, len(frames_reduced))
        if not guip.is_alive():
            break
        
def evaluate_all():
    with open("evaluation_results.txt", "a+") as f:
        f.write(f"SESSION {time()}\n")
    np.random.seed(SEED) # ඞ
    data_path = "data/"

    protocols = {
    #"P1E": ["S1", "S2", "S3", "S4"],
    #"P1L": ["S1", "S2", "S3", "S4"],
    "P2E": ["S1", "S2", "S3", "S4"],
    #"P2L": ["S1", "S2", "S3", "S4"],
    }

    #thresholds = [0.1, 0.2, 0.225, 0.25, 0.275, 0.3, 0.4, 0.5]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,  0.9]
    
    scenarios: list[tuple[str, str]] = []
    for environment in protocols: # P1E, P1L, P2E, P2L
        env_scenarios = protocols[environment]
        # We append to scenarios the corresponding folders to env_scenarios
        for scenario in env_scenarios: # S1, S2, S3, S4
            path_env_scenario = f"{environment}_{scenario}"
            if environment in ["P2E"] or environment == "P2L" and scenario == "S4":
                scenarios.append((environment, path_env_scenario, ".1"))
                scenarios.append((environment, path_env_scenario, ".2"))
            elif environment == "P2L":
                scenarios.append((environment, path_env_scenario, ".2"))
            else:
                scenarios.append((environment, path_env_scenario, ""))

    # For each environment
    for environment, scenario, suffix_scenario in scenarios:
        groundtruth_paths = [os.path.join("data/groundtruth/", f"{scenario}_C{i+1}{suffix_scenario}.xml") for i in range(MAX_CAMERAS)]
        # Build the gallery
        print("Building the gallery...")
        gallery, _ = build_gallery(groundtruth_paths)
        # For each threshold
        for threshold in thresholds:
            print(f"Environment: {environment}, Scenario: {scenario}, Threshold: {threshold}")
            # Initialize the identities
            unknown_identities: list[Identity] = [] # temporary identities which don't have a label yet
            known_identities: list[Identity] = [] # permanent identities which have a label

            # Load all frames
            print("Loading frames...")
            paths = [ os.path.join(data_path, environment, f"{scenario}_C{i+1}{suffix_scenario}") for i in range(MAX_CAMERAS)] # paths of the cameras
            print(paths, threshold)
            frames = list(filter(lambda x: x.endswith(".jpg"), os.listdir(paths[0]))) # frames of the cameras
            frames = sorted(frames)
            frames_reduced = frames[:] # frames to be processed
            # all_camera_images = [[cv2.imread(os.path.join(path, frame)) for path in paths] for frame in frames_reduced] # images of the cameras

            all_frames_no_cameras = list(map(lambda x: x.split(".")[0], frames_reduced))

            i = 0
            for frame_name in tqdm(all_frames_no_cameras):
                all_camera_image = [cv2.imread(os.path.join(path, frames[i])) for path in paths]
                handle_frame(all_camera_image, gallery, unknown_identities, known_identities, frame_name, threshold)
                i += 1

            # Force last decision
            unknown_identities, known_identities = decide_identities(unknown_identities, known_identities, gallery, threshold, force=True)

            # Evaluate the results
            all_frames_cameras = []
            for i in range(MAX_CAMERAS):
                for frame in map(lambda x: x.split(".")[0], frames_reduced):
                    all_frames_cameras.append(f"{i+1}_{frame}")
            eval_res = evaluate_system(known_identities, unknown_identities, groundtruth_paths, all_frames_cameras, gallery)

            
            with open("evaluation_results.txt", "a+") as f:
                f.write(f"SCENARIO: {environment}_{scenario}{suffix_scenario}, THRESHOLD: {threshold}\n")
                f.write(json.dumps(eval_res) + "\n")

            # Attempt to free memory
            del unknown_identities
            del known_identities
            del all_frames_no_cameras
            del all_frames_cameras
            del frames_reduced
            del frames
            gc.collect()
        del gallery
        gc.collect()

if __name__ == "__main__":
    main()
    #evaluate_all()
