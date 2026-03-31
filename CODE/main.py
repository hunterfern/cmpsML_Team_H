from io_utils import load_training_data, save_dataframe_to_excel
from preprocess import preprocess_data
from config import (
    TARGET_COLUMN,
    RAW_DATA_OUTPUT_FILE,
    PROCESSED_DATA_OUTPUT_FILE
)

def main():
    # Load raw dataset
    raw_df = load_training_data()
    print("Loaded dataset shape:", raw_df.shape)

    # Saves raw dataset to excel
    save_dataframe_to_excel(raw_df, RAW_DATA_OUTPUT_FILE)

    # Preprocess dataset
    (
        encoded_features,
        encoded_target,
        feature_encoder,
        target_encoder,
        preprocessed_df,
        cleaned_df
    ) = preprocess_data(raw_df, TARGET_COLUMN)

    print("Encoded feature matrix shape:", encoded_features.shape)
    print("Encoded target vector shape:", encoded_target.shape)
    print("Preprocessed dataset shape:", preprocessed_df.shape)

    # Saves preprocessed dataset to excel
    save_dataframe_to_excel(preprocessed_df, PROCESSED_DATA_OUTPUT_FILE)

    print("Preprocessing complete. Raw and preprocessed datasets saved to OUTPUT folder.")

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Class distribution
    sns.countplot(x=raw_df["poisonous"])
    plt.title("Poisonous vs Edible Mushrooms")
    plt.show()

    # Odor vs class
    plt.figure()
    sns.countplot(x=raw_df["odor"], hue=raw_df["poisonous"])
    plt.title("Odor vs Poisonous")
    plt.xticks(rotation=45)
    plt.show()

if __name__ == "__main__":
    main()
