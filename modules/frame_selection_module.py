import numpy as np

def select_best_frame(frames):
    if len(frames) == 1:
        return frames[0]
    else:
        # TODO: Implement your own frame selection algorithm here
        best = frames[np.random.randint(len(frames))]
        # IDEA: We could choose the frame with the highest confidence score
        # IDEA: We could choose the frame with more faces detected
        # IDEA: We could use the faces extracted from the three cameras and check if they are the same person to continuously track the same person
        # ALSOIDEA: Otherwise we could just use position inside the frame (easier but not as good (and maybe not so easy))
        # ALSOALSOIDEA: Otherwise we could just use this approach (continuously track the same person) only if 1 camera is used
        return best