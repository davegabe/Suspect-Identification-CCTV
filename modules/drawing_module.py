import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from mpl_toolkits.axes_grid1 import ImageGrid
from multiprocessing import Process, Queue
import cv2

from modules.gallery_module import Identity


def draw_files(identities: list[Identity], frames: list[str], paths: list[str]):
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
                        cv2.putText(camera_img, f"{identity.ranked_names[0]}, id:{identity.id}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                found = True
            # Save the image
            if found:
                plt.imshow(camera_img)
                plt.imsave('results/' + f"_{num_cam}_" + frame + ".png", camera_img)
                plt.close()

class GUI(Process):
    """
    Class to draw the GUI.
    """

    def __init__(self, requests_queue: Queue, responses_queue: Queue, n_frames: int, all_frames: list[str], gallery_sample: dict[str, np.ndarray]):
        super(GUI, self).__init__()
        # Create the queues
        self.requests_queue: Queue = requests_queue # using data as (frame, camera)
        self.responses_queue: Queue = responses_queue # using data as (frame, known_identities, unknown_identities)
        # Requested frame of the video and requested camera
        self.req_frame: int = 0
        self.req_camera: int = 0
        self.curr_frame: int = 0 # Current processed frame of the video
        self.n_frames: int = n_frames
        self.all_frames: list[str] = all_frames
        self.gallery_sample: dict[str, np.ndarray] = dict(sorted(gallery_sample.items()))

    def run(self):
        """
        Run the GUI.
        """
        # Identities
        self.known_identities: list[Identity] = []
        self.unknown_identities: list[Identity] = []
        # Frame of the video
        self.frame: np.ndarray = np.zeros((1, 1, 3), dtype=np.uint8)
        # Create the figure
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect("key_press_event", self.on_press)
        # Create the subplots
        self.video_ax = self.fig.add_subplot(1, 2, 1)
        self.slider_ax = self.fig.add_axes([0.1, 0.05, 0.8, 0.03])
        self.camera_buttons_ax = self.fig.add_axes([0.1, 0.1, 0.1, 0.1], facecolor="lightblue")
        # Interactive stuff
        self.slider = Slider(self.slider_ax, "Frame", 0, self.n_frames - 1, valinit=self.req_frame, valstep=1)
        self.slider.on_changed(self.update_req_frame)
        self.camera_buttons = RadioButtons(self.camera_buttons_ax, ("Camera 1", "Camera 2", "Camera 3"))
        self.camera_buttons.on_clicked(self.update_req_camera)
        # Draw the gallery
        self.draw_gallery_images_bar()
        # Launch the GUI
        self.draw_gui()
        plt.title("Suspect Identification in CCTV Footage")
        # Maximize the window and show the GUI
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

    def update_req_frame(self, val: int):
        """
        Update the requested frame of the video.

        Args:
            val (int): Index of the requested frame of the video.
        """
        # Update the requested frame if it has changed
        if val != self.req_frame:
            self.req_frame = min(int(val), self.n_frames - 1)
            # Update the GUI
            self.draw_gui()

    def update_req_camera(self, val: int):
        """
        Update the requested camera of the video.

        Args:
            val (int): Index of the requested camera of the video.
        """
        # Convert the string to an integer
        if val == "Camera 1":
            new_val = 0
        elif val == "Camera 2":
            new_val = 1
        elif val == "Camera 3":
            new_val = 2
        # Update the requested camera if it has changed
        if new_val != self.req_camera:
            self.req_camera = new_val
            # Update the GUI
            self.draw_gui()

    def on_press(self, event):
        """
        Handle the key press event.

        Args:
            event (Event): Key press event.
        """
        if event.key == "left" and self.req_frame > 0:
            self.req_frame -= 1
        elif event.key == "right" and self.req_frame < self.n_frames - 1:
            self.req_frame += 1
        # Update the GUI
        self.draw_gui()

    def draw_video(self):
        """
        Draw the video in the GUI.
        """
        # Clear the subplot
        self.video_ax.clear()
        # Disable the axis
        self.video_ax.axis("off")
        # Draw the requested frame of the video
        self.video_ax.imshow(self.frame)

    def draw_gallery_images_bar(self):
        """
        Draw a subplot with a photo of each identity in the gallery.
        """
        # Get how many identities there are in the gallery
        n_identities = len(self.gallery_sample.keys())
        # Create a grid of subplots to draw the photos of the identities
        grid = ImageGrid(self.fig, 122, nrows_ncols=(3, n_identities//3), axes_pad=0.1)
        # For each identity in the requested frame
        for i, (identity, face) in enumerate(self.gallery_sample.items()):
            # Draw the photo of the identity
            grid[i].imshow(face)
            # Disable the axis
            grid[i].axis("off")
            # Draw the name of the identity at the bottom center of the photo, with a white background
            grid[i].text(0.5, 0.1, identity, ha="center", va="center", transform=grid[i].transAxes, bbox=dict(facecolor="white", alpha=0.5))

    def draw_slider(self):
        """
        Draw a slider to change the requested frame of the video.
        """
        # Draw the slider
        self.slider.set_val(self.req_frame)

    def draw_camera_buttons(self):
        """
        Draw three toggle buttons, one for each camera.
        """
        # Draw the buttons
        self.camera_buttons.set_active(self.req_camera)

    def draw_gui(self):
        """
        Draw the GUI using subplots. The GUI will have 4 subplots:
        1. The video
        2. The bar with the photos of the gallery
        3. The slider to change the requested frame of the video
        4. Three toggle buttons, one for each camera
        """
        # Ask for the requested frame of the video
        self.ask_for_frame()
        # Draw the video
        self.draw_video()
        # Draw the slider
        self.draw_slider()
        # Draw three toggle buttons, one for each camera
        self.draw_camera_buttons()
        # Show the figure
        self.fig.canvas.draw()

    def draw_frame_identities(self):
        """
        Draw the bounding boxes of the identities in the requested frame and their names.
        """
        # For each identity in the requested frame, draw the bounding box and the name
        for identity in self.known_identities:
            # Check if the frame is in the identity and draw the bbox and the name
            for i in range(len(identity.frames)):
                if identity.frames[i] == f"{self.req_camera+1}_{self.all_frames[self.req_frame]}":
                    # Draw the bouding box in plt
                    x1 = int(identity.bboxes[i][0])
                    y1 = int(identity.bboxes[i][1])
                    x2 = int(identity.bboxes[i][2])
                    y2 = int(identity.bboxes[i][3])
                    # Draw the keypoints
                    for kp in identity.kps[i]:
                        cv2.circle(self.frame, (int(kp[0]), int(kp[1])), 1, (0, 0, 255), 1)

                    cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Print the identity reducing the size of the text to be minor than AA
                    print(identity.ranked_names)
                    cv2.putText(self.frame, f"{identity.ranked_names[0]}, id:{identity.id}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # For each identity in the requested frame, draw the bounding box and the name
        for identity in self.unknown_identities:
            # Check if the frame is in the identity and draw the bbox and the name
            for i in range(len(identity.frames)):
                if identity.frames[i] == f"{self.req_camera+1}_{self.all_frames[self.req_frame]}":
                    # Draw the bouding box in plt
                    x1 = int(identity.bboxes[i][0])
                    y1 = int(identity.bboxes[i][1])
                    x2 = int(identity.bboxes[i][2])
                    y2 = int(identity.bboxes[i][3])
                    # Draw the keypoints
                    for kp in identity.kps[i]:
                        cv2.circle(self.frame, (int(kp[0]), int(kp[1])), 1, (0, 0, 255), 1)

                    cv2.rectangle(self.frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # Print the identity reducing the size of the text to be minor than AA
                    cv2.putText(self.frame, f"{identity.ranked_names[0]}, id:{identity.id}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    def ask_for_frame(self):
        """
        Ask for the requested frame of the video.

        Args:
            queue (Queue): Queue to send the requested frame of the video.
        """
        # Send the requested frame and the requested camera to the queue
        self.requests_queue.put((self.req_frame, self.req_camera))
        # Get the requested frame of the video and the identities in it
        frame, known_identities, unknown_identities = self.responses_queue.get()
        # Update the requested frame of the video
        self.frame = frame
        # Swap the BGR frame to RGB
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        # Update the known identities
        self.known_identities = known_identities
        # Update the unknown identities
        self.unknown_identities = unknown_identities
        # Use identities to update the frame
        self.draw_frame_identities()