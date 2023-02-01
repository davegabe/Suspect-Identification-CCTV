import numpy as np

from modules.gallery_module import Identity, check_identity


def decide_identities(unknown_identities: list[Identity], known_identities: list[Identity], gallery: list[Identity], force: bool = False) -> tuple[list[Identity], list[Identity]]:
    """
    Decide the identities of the unknown identities based on the known identities.

    Args:
        unknown_identities (list): List of unknown identities.
        known_identities (list): List of known identities.

    Returns:
        unknown_identities (list): List of updated unknown identities.
        known_identities (list): List of updated known identities.
    """
    # For each unknown identity, check if it's in scene
    for unknown_identity in unknown_identities:
        # Check if it's not in scene
        if not unknown_identity.is_in_scene() or force:
            # We have to check if the face is similar to an unknown identity
            unknown_selected_faces: list[np.ndarray] = unknown_identity.get_biggest_faces()
            names_selected_faces: dict[str, int] = check_identity(gallery,unknown_selected_faces)
            # We get the name with the most occurences
            name = max(names_selected_faces, key=names_selected_faces.get)  #TODO: da ricontrollare copilot :)
            # TODO: we should check using the threshold? And what if it is under the threshold? 
            # We create a new identity with the name in known identities
            new_identity = unknown_identity # We copy the unknown identity
            new_identity.name = name # We change the id of the new identity with the name
            known_identities.append(new_identity)
    # We can finally remove the unknown identities from the list
    for unknown_identity in unknown_identities:
        # Check if it's not in scene
        if not unknown_identity.is_in_scene() or force:
            # We remove the unknown identity from the list
            unknown_identities.remove(unknown_identity)
    return unknown_identities, known_identities
