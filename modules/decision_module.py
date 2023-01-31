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
        # If it's in scene, check if it's in known identities
        if unknown_identity.is_in_scene:
            for known_identity in known_identities:
                # If it's in known identities, add it to the known identities
                if unknown_identity == known_identity:
                    known_identity.add_to_known(unknown_identity)
                    break
            # If it's not in known identities, add it to known identities
            else:
                known_identities.append(unknown_identity)
