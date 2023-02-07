TEST_PATH = "data/P1E" # Path to the test dataset
TEST_SCENARIO = "P1E_S1" # Scenario to use for the test (this must be a subfolder of TEST_PATH, but must not include the "_C*" suffix, e.g. "P1E_S1" instead of "P1E_S1_C1")
TEST_SCENARIO2 = "" # Some scenarios have two versions, when this is the case this variable specifies which one. It must include the "." prefix. Otherwise it must be blank
GALLERY_PATH = "data/gallery" # Path to the gallery dataset (this should be different from the test dataset and must be a "*_faces" folder)
GALLERY_SCENARIO = "P1L_S1_C1" # Path to the gallery dataset (this must be a subfolder of GALLEY_PATH, this must include the "_C*" suffix, e.g. "P1L_S1_C1")
MAX_CAMERAS = 3 # Maximum number of cameras to use

UNKNOWN_SIMILARITY_THRESHOLD = 0.28 # Threshold used to decide if a face is similar to an unknown identity
MAX_MISSING_FRAMES = 10 # Maximum number of frames an unknown identity can be missing before became to known identities (using decision module)
GALLERY_THRESHOLD = 0.2 # Threshold used to decide if a face is in the gallery (to decide a known identity)
MAX_GALLERY_SIZE = 15 # Maximum number of faces in the gallery
MATCH_MODALITY = "max" # "max" or "mean"
NUMBER_OF_LAST_FACES = 5 # Number of last faces to use to decide if a face is similar to an unknown identity
NUM_BEST_FACES = 5 # Number of best faces to use to decide if a face is similar to an known identity