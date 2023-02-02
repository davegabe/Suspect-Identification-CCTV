import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, RadioButtons
from multiprocessing import Process


from modules.gallery_module import Identity


class GUI(Process):
    """
    Class to draw the GUI.
    """

    def __init__(self, known_identities: list[Identity], frames: list[list[np.ndarray]]):
        super(GUI, self).__init__()
        self.curr_frame: int = 0
        self.curr_camera: int = 0
        self.frames: list[list[np.ndarray]] = frames
        self.known_identities: list[Identity] = known_identities
        # Create the figure
        self.fig = plt.figure()
        self.fig.canvas.mpl_connect("key_press_event", self.on_press)
        # Create the subplots
        self.video_ax = self.fig.add_subplot(1, 2, 1)
        self.suspects_ax = self.fig.add_subplot(1, 2, 2)
        self.slider_ax = self.fig.add_axes([0.1, 0.05, 0.8, 0.03])
        self.camera_buttons_ax = self.fig.add_axes([0.1, 0.1, 0.1, 0.1], facecolor="lightblue")
        # Interactive stuff
        self.slider = Slider(self.slider_ax, "Frame", 0, len(self.frames) - 1, valinit=self.curr_frame, valstep=1)
        self.slider.on_changed(self.update_curr_frame)
        self.camera_buttons = RadioButtons(self.camera_buttons_ax, ("Camera 1", "Camera 2", "Camera 3"))
        self.camera_buttons.on_clicked(self.update_curr_camera)
        # Launch the GUI
        # plt.ion()
        # plt.show()
        self.draw_gui()

    def update_curr_frame(self, val: int):
        """
        Update the current frame of the video.

        Args:
            val (int): Index of the current frame of the video.
        """
        # Update the current frame if it has changed
        if val != self.curr_frame:
            self.curr_frame = val
            print(self.curr_frame)
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
        self.video_ax.imshow(self.frames[self.curr_frame][self.curr_camera])

    def draw_suspect_images_bar(self):
        """
        Draw a subplot with a photo of each known identity inside the current frame and their name.
        """
        # Clear the subplot
        self.suspects_ax.clear()

        # Draw the subplot
        # For each identity in the current frame, draw a photo of the identity and its name
        for i, suspect in enumerate(self.known_identities):
            # Draw the photo of the identity
            # self.suspects_ax.imshow(suspect.photo)

            # Draw the name of the identity
            self.suspects_ax.text(0, 0, suspect.name)

        pass

    def draw_slider(self):
        """
        Draw a slider to change the current frame of the video.
        """
        # # Clear the subplot
        # self.slider_ax.clear()

        # Draw the slider
        self.slider.set_val(self.curr_frame)

    def draw_camera_buttons(self):
        """
        Draw three toggle buttons, one for each camera.
        """
        # # Clear the subplot
        # self.camera_buttons_ax.clear()

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
        # Draw the video
        self.draw_video()
        # Draw the bar with the photos of the known identities
        self.draw_suspect_images_bar()
        # Draw the slider
        self.draw_slider()
        # Draw three toggle buttons, one for each camera
        self.draw_camera_buttons()
        # Show the figure
        plt.show()

    def on_press(self, event):
        if event.key == "left" and self.curr_frame > 0:
            self.curr_frame -= 1
        elif event.key == "right" and self.curr_frame < len(self.frames) - 1:
            self.curr_frame += 1
        # Update the GUI
        self.draw_gui()

# if __name__ == "__main__":
#     # Create a list of known identities
#     known_identities = [
#         Identity(),
#     ]

#     # Create a list of frames
#     frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(10)]

#     # Run the main function
#     GUI().start()
