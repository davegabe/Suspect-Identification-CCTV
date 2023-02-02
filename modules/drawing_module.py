import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from mpl_toolkits.axes_grid1 import ImageGrid
from multiprocessing import Process, Queue
import cv2

from modules.gallery_module import Identity


class GUI(Process):
    """
    Class to draw the GUI.
    """

    def __init__(self, requests_queue: Queue, responses_queue: Queue, n_frames: int, gallery_path: str):
        super(GUI, self).__init__()
        # Create the queues
        self.requests_queue: Queue = requests_queue # using data as (frame, camera)
        self.responses_queue: Queue = responses_queue # using data as (frame, known_identities, unknown_identities)
        # Current frame of the video and current camera
        self.curr_frame: int = 0
        self.curr_camera: int = 0
        self.n_frames: int = n_frames
        # Gallery path
        self.gallery_path: str = gallery_path + "_faces"

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
        self.suspects_ax = self.fig.add_subplot(1, 2, 2)
        self.slider_ax = self.fig.add_axes([0.1, 0.05, 0.8, 0.03])
        self.camera_buttons_ax = self.fig.add_axes([0.1, 0.1, 0.1, 0.1], facecolor="lightblue")
        # Interactive stuff
        self.slider = Slider(self.slider_ax, "Frame", 0, self.n_frames - 1, valinit=self.curr_frame, valstep=1)
        self.slider.on_changed(self.update_curr_frame)
        self.camera_buttons = RadioButtons(self.camera_buttons_ax, ("Camera 1", "Camera 2", "Camera 3"))
        self.camera_buttons.on_clicked(self.update_curr_camera)
        # Pick random faces from the gallery to show in the GUI
        self.faces: dict[str, np.ndarray] = {}
        path = os.path.join(self.gallery_path, os.listdir(self.gallery_path)[0])
        names = os.listdir(path)
        # For each name in the gallery
        for name in names:
            # Path of the face
            images = os.listdir(os.path.join(path, name))
            # Pick a random image of the face
            image = images[np.random.randint(0, len(images))]
            # Read the image
            self.faces[name] = cv2.imread(os.path.join(path, name, image))
        # Launch the GUI
        self.draw_gui()
        plt.show()

    def update_curr_frame(self, val: int):
        """
        Update the current frame of the video.

        Args:
            val (int): Index of the current frame of the video.
        """
        # Update the current frame if it has changed
        if val != self.curr_frame:
            self.curr_frame = min(int(val), self.n_frames - 1)
            # Update the GUI
            self.draw_gui()

    def update_curr_camera(self, val: int):
        """
        Update the current camera of the video.

        Args:
            val (int): Index of the current camera of the video.
        """
        # Convert the string to an integer
        if val == "Camera 1":
            new_val = 0
        elif val == "Camera 2":
            new_val = 1
        elif val == "Camera 3":
            new_val = 2
        # Update the current camera if it has changed
        if new_val != self.curr_camera:
            self.curr_camera = new_val
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
        # Draw the current frame of the video
        self.video_ax.imshow(self.frame)

    def draw_suspect_images_bar(self):
        """
        Draw a subplot with a photo of each known identity inside the current frame and their name.
        """
        # Clear the subplot
        self.suspects_ax.clear()
        # Disable the axis
        self.suspects_ax.axis("off")
        # Get how many identities there are in the current frame
        identities_in_frame = [identity for identity in self.known_identities if f"{self.curr_camera}_{self.curr_frame}" in identity.frames]
        n_identities = len(identities_in_frame)
        if n_identities > 0:
            # Create a new figure
            fig = plt.figure()
            # Create a grid of subplots to draw the photos of the identities
            grid = ImageGrid(fig, 111, nrows_ncols=(1, n_identities), axes_pad=0.1)
            # For each identity in the current frame
            for i, identity in enumerate(identities_in_frame):
                # Get the name of the identity
                name = identity.name
                # Get the photo of the identity
                face = self.faces[name]
                # Draw the photo of the identity
                grid[i].imshow(face)
                # Disable the axis
                grid[i].axis("off")
                # Draw the name of the identity
                grid[i].text(0, 0, name, color="white", bbox=dict(facecolor="black", alpha=0.5))
            # Show the figure in the subplot
            fig.canvas.draw()
            self.suspects_ax.imshow(fig.canvas.buffer_rgba())
            fig.clear()


    def draw_slider(self):
        """
        Draw a slider to change the current frame of the video.
        """
        # Draw the slider
        self.slider.set_val(self.curr_frame)

    def draw_camera_buttons(self):
        """
        Draw three toggle buttons, one for each camera.
        """
        # Draw the buttons
        self.camera_buttons.set_active(self.curr_camera)

    def draw_gui(self):
        """
        Draw the GUI using subplots. The GUI will have 4 subplots:
        1. The video
        2. The bar with the photos of the known identities
        3. The slider to change the current frame of the video
        4. Three toggle buttons, one for each camera
        """
        # # Ask for the current frame of the video
        self.ask_for_frame()
        # Draw the video
        self.draw_video()
        # Draw the bar with the photos of the known identities
        self.draw_suspect_images_bar()
        # Draw the slider
        self.draw_slider()
        # Draw three toggle buttons, one for each camera
        self.draw_camera_buttons()
        # Show the figure
        self.fig.canvas.draw()

    def on_press(self, event):
        """
        Handle the key press event.

        Args:
            event (Event): Key press event.
        """
        if event.key == "left" and self.curr_frame > 0:
            self.curr_frame -= 1
        elif event.key == "right" and self.curr_frame < self.n_frames - 1:
            self.curr_frame += 1
        # Update the GUI
        self.draw_gui()

    def draw_frame_identities(self):
        """
        Draw the bounding boxes of the identities in the current frame and their names.
        """
        # For each identity in the current frame, draw the bounding box and the name
        for identity in self.known_identities:
            # Check if the frame is in the identity and draw the bbox and the name
            for i in range(len(identity.frames)):
                if identity.frames[i] == f"{self.curr_camera}_{self.curr_frame}":
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
                    cv2.putText(self.frame, f"{identity.name}, id:{identity.id}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # For each identity in the current frame, draw the bounding box and the name
        for identity in self.unknown_identities:
            # Check if the frame is in the identity and draw the bbox and the name
            for i in range(len(identity.frames)):
                if identity.frames[i] == f"{self.curr_camera}_{self.curr_frame}":
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
                    cv2.putText(self.frame, f"{identity.name}, id:{identity.id}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    def ask_for_frame(self):
        """
        Ask for the current frame of the video.

        Args:
            queue (Queue): Queue to send the current frame of the video.
        """
        # Send the current frame and the current camera to the queue
        self.requests_queue.put((self.curr_frame, self.curr_camera))
        # Get the current frame of the video and the identities in it
        frame, known_identities, unknown_identities = self.responses_queue.get()
        # Update the current frame of the video
        self.frame = frame
        # Update the known identities
        self.known_identities = known_identities
        # Update the unknown identities
        self.unknown_identities = unknown_identities
        # Use identities to update the frame
        self.draw_frame_identities()