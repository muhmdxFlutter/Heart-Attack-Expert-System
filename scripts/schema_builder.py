import pandas as pd
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(BASE_DIR, 'data', 'heart_2022_no_nans.csv')
DEFAULT_OUT = os.path.join(BASE_DIR, 'models', 'model_schema.json')

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

    # 2. Get 121 encoded features to define the expected model inputs
    cat_cols = x_raw.select_dtypes(exclude=[np.number]).columns.tolist()
    x_encoded = pd.get_dummies(x_raw, columns=cat_cols, drop_first=True, dtype=int)
    all_features = x_encoded.columns.tolist()

    # 3. Compute baseline values (median for numerical, mode for categorical)
    # This prevents sending '0' for height, weight, sleep time which ruins the Standard Scaler.
    num_cols = x_raw.select_dtypes(include=[np.number]).columns.tolist()
    
    baselines = {}
    for col in num_cols:
        baselines[col] = float(x_raw[col].median())
        
    for col in cat_cols:
        baselines[col] = str(x_raw[col].mode()[0])

    schema = {
        'expected_features': all_features,
        'numerical_features': num_cols,
        'categorical_features': cat_cols,
        'baselines': baselines
    }
    
    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=4)
        
    print(f"Schema successfully saved to {output_path}")
    print(f"Total Features Expected by Model: {len(all_features)}")
    print("Baseline (Average Patient) values extracted and saved.")
    
if __name__ == "__main__":
    build_schema()
