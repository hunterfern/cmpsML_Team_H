import pandas as pd
from config import TARGET_COLUMN

def extract_features(preprocessed_df):
    extracted_df = preprocessed_df.copy()
    
    # 1. Frequency encoding
    # Odor frequency
    odor_cols = [col for col in extracted_df.columns if col.startswith('odor_')]
    if odor_cols:
        odor_counts = extracted_df[odor_cols].sum() / len(extracted_df)
        extracted_df['odor_freq'] = (extracted_df[odor_cols] * odor_counts).sum(axis=1)
    
    # Gill-color frequency
    gill_cols = [col for col in extracted_df.columns if col.startswith('gill-color_')]
    if gill_cols:
        gill_counts = extracted_df[gill_cols].sum() / len(extracted_df)
        extracted_df['gill_color_freq'] = (extracted_df[gill_cols] * gill_counts).sum(axis=1)
    
    # Spore-print-color frequency
    spore_cols = [col for col in extracted_df.columns if col.startswith('spore-print-color_')]
    if spore_cols:
        spore_counts = extracted_df[spore_cols].sum() / len(extracted_df)
        extracted_df['spore_print_freq'] = (extracted_df[spore_cols] * spore_counts).sum(axis=1)
    
    # 2. Important interactions
    bruises_cols = [col for col in extracted_df.columns if 'bruises' in col]
    odor_none_cols = [col for col in extracted_df.columns if col.startswith('odor_') and col.endswith('_n')]
    
    if bruises_cols and odor_none_cols:
        extracted_df['bruises_no_odor'] = (extracted_df[bruises_cols[0]] * extracted_df[odor_none_cols[0]]).astype(int)
    
    # 3. Suspicious combination
    spore_risk_cols = [col for col in extracted_df.columns if 'spore-print-color' in col and (col.endswith('_h') or col.endswith('_w'))]
    if odor_none_cols and spore_risk_cols:
        extracted_df['suspicious_spore_gill'] = (extracted_df[odor_none_cols[0]] * extracted_df[spore_risk_cols].max(axis=1)).astype(int)
    
    # 4. Stalk root missing
    stalk_missing_cols = [col for col in extracted_df.columns if 'stalk-root' in col and ('missing' in col.lower() or '?' in col or 'nan' in col.lower())]
    if stalk_missing_cols:
        extracted_df['stalk_root_missing'] = extracted_df[stalk_missing_cols[0]]
    
    return extracted_df
