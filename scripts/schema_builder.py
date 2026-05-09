import pandas as pd
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(BASE_DIR, 'data', 'heart_2022_no_nans.csv')
DEFAULT_OUT = os.path.join(BASE_DIR, 'models', 'model_schema.json')

# Must match heart_disease_smote.py exactly so the schema feature count is consistent
NOISE_FEATURES = [
    # --- killers 8 ---
    'State',
    'HIVTesting',
    'FluVaxLast12',
    'PneumoVaxEver',
    'TetanusLast10Tdap',
    'DeafOrHardOfHearing',
    'BlindOrVisionDifficulty',
    'RaceEthnicityCategory',
    # --- 3 low-signal proxies ---
    'HadSkinCancer',
    'RemovedTeeth',
    'HighRiskLastYear',
    # --- 3 reverse-causation confounders ---
    'AlcoholDrinkers',
    'CovidPos',
    'HadAsthma',
]

# Ordinal mappings — must mirror heart_disease_smote.py exactly
SMOKER_ORDINAL = {
    'Never smoked': 0,
    'Former smoker': 1,
    'Current smoker - now smokes some days': 2,
    'Current smoker - now smokes every day': 3,
}
ECIG_ORDINAL = {
    'Never used e-cigarettes in my entire life': 0,
    'Not at all (right now)': 1,
    'Use them some days': 2,
    'Use them every day': 3,
}
GEN_HEALTH_ORDINAL = {
    'Excellent': 0,
    'Very good': 1,
    'Good':      2,
    'Fair':      3,
    'Poor':      4,
}

def build_schema(csv_path=DEFAULT_CSV, output_path=DEFAULT_OUT):
    print(f"Loading '{csv_path}' to extract model schema...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Please ensure the dataset is in the current directory.")
        return
    
    # 1. Target removal
    if 'HadHeartAttack' in df.columns:
        x_raw = df.drop(columns=['HadHeartAttack'])
    else:
        x_raw = df.copy()

    # 1b. Drop the same 8 noise features removed during training
    #     Without this, schema_builder generates 121 features while the
    #     trained model expects only 56 — causing a shape mismatch at inference.
    existing_noise = [f for f in NOISE_FEATURES if f in x_raw.columns]
    x_raw = x_raw.drop(columns=existing_noise)
    print(f'Dropped {len(existing_noise)} noise features to match training pipeline.')

    # 2. One-hot encode categorical columns (identical flags to training)
    cat_cols = x_raw.select_dtypes(exclude=[np.number]).columns.tolist()

    # Apply ordinal encoding to match training pipeline
    x_raw['SmokerStatus']    = x_raw['SmokerStatus'].map(SMOKER_ORDINAL).fillna(0).astype(int)
    x_raw['ECigaretteUsage'] = x_raw['ECigaretteUsage'].map(ECIG_ORDINAL).fillna(0).astype(int)
    x_raw['GeneralHealth']   = x_raw['GeneralHealth'].map(GEN_HEALTH_ORDINAL).fillna(0).astype(int)
    cat_cols = [c for c in cat_cols if c not in ('SmokerStatus', 'ECigaretteUsage', 'GeneralHealth')]

    x_encoded = pd.get_dummies(x_raw, columns=cat_cols, drop_first=True, dtype=int)
    all_features  = x_encoded.columns.tolist()
    num_cols      = x_raw.select_dtypes(include=[np.number]).columns.tolist()
    binary_cols   = [c for c in all_features if c not in num_cols]

    # 3. Compute baseline values (median for numerical, mode for categorical)
    # This prevents sending ‘0’ for height, weight, sleep time which ruins the Standard Scaler.
    baselines = {}
    for col in num_cols:
        baselines[col] = float(x_raw[col].median())

    for col in cat_cols:
        baselines[col] = str(x_raw[col].mode()[0])

    schema = {
        'expected_features':   all_features,
        'numerical_features':  num_cols,      # scaled by ColumnTransformer (incl. ordinals)
        'binary_features':     binary_cols,   # passthrough (already in [0, 1])
        'categorical_features': cat_cols,
        'ordinal_mappings': {
            'SmokerStatus':    SMOKER_ORDINAL,
            'ECigaretteUsage': ECIG_ORDINAL,
            'GeneralHealth':   GEN_HEALTH_ORDINAL,
        },
        'baselines': baselines
    }
    
    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=4)
        
    print(f"Schema successfully saved to {output_path}")
    print(f"Total Features Expected by Model: {len(all_features)}")
    print("Baseline (Average Patient) values extracted and saved.")
    
if __name__ == "__main__":
    build_schema()
