from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
INPUT_DIR = BASE_DIR / "INPUT"
TRAIN_DIR = INPUT_DIR / "TRAIN"
TEST_DIR = INPUT_DIR / "TEST"

OUTPUT_DIR = BASE_DIR / "OUTPUT"
MODEL_DIR = BASE_DIR / "MODEL"
FIGURE_DIR = BASE_DIR / "DOC" / "figures"

# File name (where datset goes)
DATA_FILE = "mushroom.csv"

# Target column
TARGET_COLUMN = "poisonous"

# Ouput file names
RAW_DATA_OUTPUT_FILE = "raw_data.xlsx"
PROCESSED_DATA_OUTPUT_FILE = "processed_data.xlsx"
