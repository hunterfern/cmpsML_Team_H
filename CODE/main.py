from io_utils import load_training_data, save_dataframe_to_excel
from preprocess import preprocess_data
from feature_extraction import extract_features
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

    # Feature Extraction
    extracted_df = extract_features(preprocessed_df)
    
    print("Extracted features shape:", extracted_df.shape)

    # Save extracted features
    save_dataframe_to_excel(extracted_df, "extracted_features.xlsx")

    print("Feature extraction complete. Extracted data saved to OUTPUT/extracted_features.xlsx")

    # Plots
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
    plt.title("Distribution of Odor by Class")
    plt.xlabel("Odor")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="Class")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "odor_vs_class.png")
    plt.show()

    # Correlation heatmap
    numeric_df = extracted_df.select_dtypes(include='number')
    plt.figure(figsize=(16, 12))
    corr = numeric_df.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, linewidths=0.3, square=True, 
                cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Heatmap After Feature Extraction")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_correlation_heatmap.png")
    plt.show()

    odor_freq_map = raw_df['odor'].value_counts(normalize=True)
    extracted_df['odor_freq'] = raw_df['odor'].map(odor_freq_map).values

    # Violin plot - Odor Frequency by Class
    extracted_df['class_labels'] = class_labels.values
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=extracted_df, x="class_labels", y="odor_freq", hue="class_labels", palette=["#2ca02c", "#d62728"], legend=False)
    plt.title("Odor Frequency by Class")
    plt.xlabel("Mushroom Class")
    plt.ylabel("Odor Frequency")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "odor_freq_by_class.png")
    plt.show()


    # Distribution of Odor Frequency by Class
    plt.figure(figsize=(8, 5))
    sns.histplot(data=extracted_df, x="odor_freq", hue=class_labels, kde=True, bins=25, 
                 palette=["#2ca02c", "#d62728"], multiple="stack")
    plt.title("Distribution of Odor Frequency by Class")
    plt.xlabel("Odor Frequency")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "odor_freq_distribution.png")
    plt.show()
    plt.close()

    print("All plots saved to figures folder.")

if __name__ == "__main__":
    main()
