from modules.gallery_module import Identity


def decide_identities(known_identities: list[Identity], unknown_identities: list[Identity]):
    """
    Decide the identities of the unknown identities based on the known identities.

    Args:
        known_identities (list): List of known identities.
        unknown_identities (list): List of unknown identities.
    """
    # For each unknown identity, check if it's in scene
    for unknown_identity in unknown_identities:
        # Check if it's not in scene
        if not unknown_identity.is_in_scene():
            # TODO: Decide the identity of the unknown identity and add it to the known identities
            pass
