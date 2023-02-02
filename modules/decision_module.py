import numpy as np

from modules.gallery_module import Identity, check_identity


def decide_identities(unknown_identities: list[Identity], known_identities: list[Identity], gallery: list[Identity], force: bool = False) -> tuple[list[Identity], list[Identity]]:
    """
    Decide the identities of the unknown identities based on the known identities.

    Args:
        unknown_identities (list): List of unknown identities.
        known_identities (list): List of known identities.
        gallery (list): List of identities in the gallery.
        force (bool, optional): Force the decision. Defaults to False.

    Returns:
        unknown_identities (list): List of updated unknown identities.
        known_identities (list): List of updated known identities.
    """
    identified_identities: list[int] = []
    # For each unknown identity, check if it's in scene
    for i, unknown_identity in enumerate(unknown_identities):
        # Check if it's not in scene
        if not unknown_identity.is_in_scene() or force:
            # We have to check if the face is similar to an unknown identity
            unknown_selected_faces: list[np.ndarray] = unknown_identity.get_biggest_faces()
            names_selected_faces: dict[str, int] = check_identity(gallery,unknown_selected_faces)
            # If there are no similar faces above the threshold, we keep the identity as unknown
            if len(names_selected_faces.keys()) == 0:
                continue
            # We get the name with the most occurences
            name = max(names_selected_faces, key=names_selected_faces.get)  #TODO: da ricontrollare copilot :)
            # TODO: we should check using the threshold? And what if it is under the threshold? 
            # We create a new identity with the name in known identities
            new_identity = unknown_identity # We copy the unknown identity
            new_identity.name = name # We change the id of the new identity with the name
            known_identities.append(new_identity)
            identified_identities.append(i)
    # We can finally remove the unknown identities from the list
    for i in reversed(identified_identities):
        unknown_identities.pop(i)
    return unknown_identities, known_identities
