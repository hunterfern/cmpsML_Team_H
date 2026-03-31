import pandas as pd
from config import TRAIN_DIR, DATA_FILE, OUTPUT_DIR

# Column names from datset documentation
COLUMN_NAMES = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
    "poisonous"
]

def load_training_data():
    file_path = TRAIN_DIR / DATA_FILE

    df = pd.read_csv(file_path)
    print("Loaded dataset shape:", df.shape)
    df.columns = COLUMN_NAMES

    return df

def save_dataframe_to_excel(df, file_name):
    # Save DataFrame to Excel in OUTPUT folder
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / file_name
    df.to_excel(output_path, index=False)
