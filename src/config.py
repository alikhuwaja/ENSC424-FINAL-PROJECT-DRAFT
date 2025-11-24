from pathlib import Path

# -----------------------------
#   PATHS
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

RAVDESS_CSV = DATA_DIR / "ravdess_index.csv"
CREMAD_CSV = DATA_DIR / "cremad_index.csv"
RAVDESS_VIDEO_CSV = DATA_DIR / "ravdess_video_index.csv"

# -----------------------------
#   AUDIO & FEATURES
# -----------------------------
SAMPLE_RATE = 16000
NUM_MEL = 64

NUM_FFT = 1024        # FFT window size
HOP_LENGTH = 512      # Hop size between frames
SEGMENT_SEC = 3       # Duration of each audio segment

# -----------------------------
#   DATASET / TRAINING PARAMS
# -----------------------------
NUM_CLASSES = 6
BATCH_SIZE = 32
N_EPOCHS = 20
LEARNING_RATE = 1e-3

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

# -----------------------------
#   MODEL SELECTION
# -----------------------------
# "crnn" or "transformer"
MODEL_TYPE = "crnn"

# -----------------------------
#   TRANSFORMER HYPERPARAMETERS
# -----------------------------
TRANSFORMER_EMBED_DIM = 128
TRANSFORMER_NUM_HEADS = 4
TRANSFORMER_FF_DIM = 256
TRANSFORMER_NUM_LAYERS = 3

# -----------------------------
#   DATA AUGMENTATION OPTIONS
# -----------------------------
USE_AUGMENTATIONS = True     # master ON/OFF switch

AUG_NOISE = True             # Gaussian noise
AUG_FREQ_MASK = True         # frequency masking
AUG_TIME_MASK = True         # time masking