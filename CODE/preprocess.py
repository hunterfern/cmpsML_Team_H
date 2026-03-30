import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def preprocess_data(df, target_column):
    # Create a copy to avoid modifying original
    processed_df = df.copy()

    # Replace missing values with readable category
    processed_df = processed_df.replace("?", "missing")
    
    # Split features and target
    feature_data = processed_df.drop(columns=[target_column])
    target_data = processed_df[target_column]

    # Encode target column into numeric labels
    target_encoder = LabelEncoder()
    encoded_target = target_encoder.fit_transform(target_data)

    # One-hot encode categorical features
    feature_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_features = feature_encoder.fit_transform(feature_data)

    # Get readable feature names after one-hot encoding
    encoded_feature_names = feature_encoder.get_feature_names_out(feature_data.columns)

    # Convert encoded features into df for export
    encoded_feature_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)

    # adds target column back to preprocessed dataset
    encoded_feature_df[target_column] = encoded_target

    return (
        encoded_features,
        encoded_target,
        feature_encoder,
        target_encoder,
        encoded_feature_df,
        processed_df
    )