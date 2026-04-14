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
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.feature_selection import mutual_info_classif

    # Readable class labels
    class_labels = raw_df["poisonous"].replace({"e": "Edible", "p": "Poisonous"})
    extracted_df['Class'] = class_labels.values

    # Readable odor labels
    odor_labels = raw_df["odor"].replace({
        "a": "Almond", "l": "Anise", "c": "Creosote", "y": "Fishy",
        "f": "Foul", "m": "Musty", "n": "None", "p": "Pungent", "s": "Spicy"
    })

    # Class distribution
    plt.figure(figsize=(8,5))
    sns.countplot(x=class_labels, order=["Edible", "Poisonous"], palette=["#2ca02c", "#d62728"])
    plt.title("Class Distributions")
    plt.xlabel("Mushroom Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "class_distribution.png")
    plt.show()

    # Odor vs class
    plt.figure(figsize=(10,6))
    odor_order = odor_labels.value_counts().index
    sns.countplot(x=odor_labels, hue=class_labels, order=odor_order, palette=["#2ca02c", "#d62728"])
    plt.title("Distribution of Odor by Class")
    plt.xlabel("Odor")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="Class")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "odor_vs_class.png")
    plt.show()

    # Gill color frequency
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=extracted_df, x="Class", y="gill_color_freq", hue="Class", palette=["#2ca02c", "#d62728"], inner="stick", legend=False)
    plt.title("Gill Color Frequency by Class")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "gill_color_freq.png")
    plt.show()

    # Spore print frequency
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=extracted_df, x="Class", y="spore_print_freq", hue="Class", palette=["#2ca02c", "#d62728"], inner="stick", legend=False)
    plt.title("Spore Print Frequency by Class")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "spore_print_freq.png")
    plt.show()

    # Bruises no odor interaction
    plt.figure(figsize=(8, 5))
    sns.countplot(data=extracted_df, x="bruises_no_odor", hue="Class", palette=["#2ca02c", "#d62728"])
    plt.title("Bruises with No Odor Interaction")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "bruises_no_odor.png")
    plt.show()

    # Suspicious spore/gill combination
    plt.figure(figsize=(8, 5))
    sns.countplot(data=extracted_df, x="suspicious_spore_gill", hue="Class", palette=["#2ca02c", "#d62728"])
    plt.title("Suspicious Spore/Gill Combination")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "suspicious_combo.png")
    plt.show()

    # Stalk root missing
    if 'stalk_root_missing' in extracted_df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=extracted_df, x="stalk_root_missing", hue="Class", palette=["#2ca02c", "#d62728"])
        plt.title("Stalk Root Missing by Class")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "stalk_root_missing.png")
        plt.show()
    
        

    # Feature impact analysis
    target_binary = raw_df["poisonous"].map({"e": 0, "p": 1})
    new_cols = ['odor_freq', 'gill_color_freq', 'spore_print_freq', 'bruises_no_odor', 'suspicious_spore_gill', 'stalk_root_missing']
    existing_cols = [c for c in new_cols if c in extracted_df.columns]
    
    mi_scores = mutual_info_classif(extracted_df[existing_cols], target_binary, random_state=42)
    mi_results = pd.Series(mi_scores, index=existing_cols).sort_values()

    plt.figure(figsize=(10, 6))
    mi_results.plot(kind='barh', color='teal')
    plt.title("Predictive Power of Extracted Features")
    plt.xlabel("Information Gain Score")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_impact.png")
    plt.show()


    print("All plots saved to figures folder.")

if __name__ == "__main__":
    main()
