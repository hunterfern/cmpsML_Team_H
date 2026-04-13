import pandas as pd
from config import TARGET_COLUMN

def extract_features(preprocessed_df):
    extracted_df = preprocessed_df.copy()
    
    # 1. Frequency encoding
    # Odor frequency
    odor_cols = [col for col in extracted_df.columns if col.startswith('odor_')]
    if odor_cols:
        extracted_df['odor_freq'] = extracted_df[odor_cols].sum(axis=1) / len(odor_cols)
    
    # Gill-color frequency
    gill_cols = [col for col in extracted_df.columns if col.startswith('gill-color_')]
    if gill_cols:
        extracted_df['gill_color_freq'] = extracted_df[gill_cols].sum(axis=1) / len(gill_cols)
    
    # Spore-print-color frequency
    spore_cols = [col for col in extracted_df.columns if col.startswith('spore-print-color_')]
    if spore_cols:
        extracted_df['spore_print_freq'] = extracted_df[spore_cols].sum(axis=1) / len(spore_cols)
    
    # 2. Important interactions
    bruises_cols = [col for col in extracted_df.columns if 'bruises' in col]
    odor_none_cols = [col for col in extracted_df.columns if col.startswith('odor_') and col.endswith('_n')]
    
    if bruises_cols and odor_none_cols:
        extracted_df['bruises_no_odor'] = extracted_df[bruises_cols[0]] * extracted_df[odor_none_cols[0]]
    
    # 3. Suspicious combination
    extracted_df['suspicious_spore_gill'] = (
        (extracted_df.filter(like='spore-print-color').sum(axis=1) > 0) &
        (extracted_df.filter(like='gill-color').sum(axis=1) > 0)
    ).astype(int)
    
    # 4. Stalk root missing
    stalk_missing_cols = [col for col in extracted_df.columns if 'stalk-root' in col and 'missing' in col.lower()]
    if stalk_missing_cols:
        extracted_df['stalk_root_missing'] = extracted_df[stalk_missing_cols[0]]
    
    return extracted_df
