from io_utils import load_training_data, save_dataframe_to_excel
from preprocess import preprocess_data
from config import (
    TARGET_COLUMN,
    RAW_DATA_OUTPUT_FILE,
    PROCESSED_DATA_OUTPUT_FILE,
    FIGURE_DIR
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

    # Readable class labels
    class_labels = raw_df["poisonous"].replace({"e": "Edible", "p": "Poisonous"})

    # Readable odor labels
    odor_labels = raw_df["odor"].replace({
        "a": "Almond",
        "l": "Anise",
        "c": "Creosote",
        "y": "Fishy",
        "f": "Foul",
        "m": "Musty",
        "n": "None",
        "p": "Pungent",
        "s": "Spicy"
    })

    # Class distribution
    plt.figure(figsize=(8,5))
    sns.countplot(x=class_labels, order=["Edible", "Poisonous"])
    plt.title("Class Distributions")
    plt.xlabel("Mushroom Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "class_distribution.png")
    plt.show()

    # Odor vs class
    plt.figure(figsize=(10,6))
    order = odor_labels.value_counts().index
    sns.countplot(x=odor_labels, hue=class_labels, order=order, palette=["#d62728", "#2ca02c"])
    plt.title("Distribution of Odor by Mushroom Class")
    plt.xlabel("Odor")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="Class")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "odor_vs_class.png")
    plt.show()

if __name__ == "__main__":
    main()
