import json
import pandas as pd
import os

#//* MODULE: inference_mapper.py
#//*   Bridges raw Streamlit UI inputs to the pruned model feature space.
#//*   After dropping 8 noise features from the training pipeline, the
#//*   expected one-hot feature count is ~110-115 (exact count determined
#//*   at training time and stored in model_schema.json).
#//*   This module is intentionally schema-driven: the JSON file is the
#//*   single source of truth — no hardcoded column lists live here.

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SCHEMA_PATH = os.path.join(BASE_DIR, 'models', 'model_schema.json')

def load_schema(schema_path=DEFAULT_SCHEMA_PATH):
    """Loads the pre-calculated schema and default baseline values."""
    with open(schema_path, 'r') as f:
        return json.load(f)

def prepare_patient_vector(user_inputs, schema, scaler):
    """
    Transforms UI inputs into the exact feature array expected by the model.

    After the Advanced Data Pruning step (8 noise features removed), the
    schema contains ~110-115 one-hot columns (exact count from model_schema.json).
    Missing inputs are intelligently filled using the dataset’s median/mode
    (stored as 'baselines' in the schema).

    Args:
        user_inputs (dict): Raw column names → values from the UI.
                            e.g. {'AgeCategory': 'Age 40 to 44',
                                  'BMI': 25.0, 'Sex': 'Male'}
        schema (dict): Loaded from model_schema.json.
        scaler: Fitted ColumnTransformer from scaler.joblib.
                Scales only the numerical columns; binary one-hot columns
                pass through unchanged (already in [0, 1]).

    Returns:
        scaled_input (np.ndarray): Shape (1, N) ready for model.predict().
        input_df (pd.DataFrame): The N-feature DataFrame for debugging.
    """
    # 1. Start with the baseline “healthy/average patient” profile (medians and modes)
    patient_raw = schema['baselines'].copy()

    # 2. Override baseline features with any explicit User inputs
    for key, value in user_inputs.items():
        if key in patient_raw:
            patient_raw[key] = value

    # 2b. Apply ordinal mappings (SmokerStatus, ECigaretteUsage)
    #//? The UI sends human-readable strings; the model expects integer ordinals
    #//? (0–3) because these features were ordinally encoded during training.
    #//? The mapping is stored in model_schema.json so no hardcoding here.
    ordinal_mappings = schema.get('ordinal_mappings', {})
    for col, mapping in ordinal_mappings.items():
        if col in patient_raw:
            val = patient_raw[col]
            # Accept either a string label or an already-converted int
            patient_raw[col] = mapping.get(str(val), int(val) if str(val).isdigit() else 0)

    # 3. Create the N-column DataFrame initialised to exactly 0.0
    #//? drop_first=True during training means the first dummy of each
    #//? categorical group is the implicit base (all zeros = base category).
    #//? Keeping every column at 0 correctly encodes the base category.
    expected_features = schema['expected_features']
    input_df = pd.DataFrame(0.0, index=[0], columns=expected_features)
    
    # 4. Populate Numerical Features
    for col in schema['numerical_features']:
        if col in expected_features:
            input_df.at[0, col] = float(patient_raw[col])
            
    # 5. Populate Categorical Features
    for col in schema['categorical_features']:
        val = str(patient_raw[col])
        dummy_col_name = f"{col}_{val}"
        
        # If the generated dummy column exists in the expected 121 features, set it to 1.
        # If it DOES NOT exist, it means it's the base category that was dropped by drop_first=True.
        # By keeping all dummy columns for this feature as 0, we correctly encode the base category!
        if dummy_col_name in expected_features:
            input_df.at[0, dummy_col_name] = 1
            
    # 6. Apply ColumnTransformer
    #//? The CT scales only numerical columns and passes binary one-hot
    #//? columns through unchanged. It accepts a DataFrame with the original
    #//? column names and handles internal column reordering automatically.
    #//! Supply the full expected_features DataFrame — do NOT reorder columns
    #//! manually; the CT uses column names, not positional indices.
    input_scaled = scaler.transform(input_df[expected_features])
    
    return input_scaled, input_df

# ==========================================
# EXAMPLE OF HOW TO INTEGRATE IN APP.PY
# ==========================================
if __name__ == "__main__":
    import numpy as np  # only needed for the self-test scaler mock below
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    print("Testing functionality...")
    try:
        schema = load_schema()
        num_cols = schema['numerical_features']
        n_features = len(schema['expected_features'])

        # Build a pass-through ColumnTransformer that mimics the saved scaler
        # (mean=0, std=1 → identity transform) just to validate shapes.
        mock_ct = ColumnTransformer(
            transformers=[('num', StandardScaler(), num_cols)],
            remainder='passthrough'
        )
        # Fit on a dummy zero row so the CT has fitted attributes
        dummy_df = pd.DataFrame(0.0, index=[0], columns=schema['expected_features'])
        mock_ct.fit(dummy_df)

        mock_ui_inputs = {
            'AgeCategory': 'Age 50 to 54',
            'Sex': 'Male',
            'BMI': 28.5,
            'GeneralHealth': 'Fair',
            'SmokerStatus': 'Former smoker',
            'HadStroke': 'No',
            'HadDiabetes': 'No'
        }

        input_scaled, raw_processed_df = prepare_patient_vector(mock_ui_inputs, schema, mock_ct)

        print(f"\n✅ Success! Generated vector shape: {input_scaled.shape}")
        print(f"Expected feature count: {n_features} (post noise-pruning)")
        print("Non-zero features:", (raw_processed_df != 0).sum(axis=1).values[0])
    except FileNotFoundError:
        print("Please run `python heart_disease_smote.py` first to regenerate model_schema.json.")
