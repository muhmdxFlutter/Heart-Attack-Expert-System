import json
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SCHEMA_PATH = os.path.join(BASE_DIR, 'models', 'model_schema.json')

def load_schema(schema_path=DEFAULT_SCHEMA_PATH):
    """Loads the pre-calculated schema and default baseline values."""
    with open(schema_path, 'r') as f:
        return json.load(f)

def prepare_patient_vector(user_inputs, schema, scaler):
    """
    Transforms UI inputs into the exact 121-column array expected by the model.
    Missing inputs are intelligently filled using the dataset's median/mode (baseline).
    
    Args:
        user_inputs: dict of raw column names -> values 
                     (e.g., {'AgeCategory': 'Age 40 to 44', 'BMI': 25.0, 'Sex': 'Male', 'HadDiabetes': 'No'})
        schema: dictionary loaded from model_schema.json
        scaler: the loaded standard scaler object
        
    Returns:
        scaled_input: numpy array of shape (1, 121) ready for model.predict()
        input_df: The 121-feature DataFrame (for debugging/verification)
    """
    # 1. Start with the baseline "healthy/average patient" profile (medians and modes)
    patient_raw = schema['baselines'].copy()
    
    # 2. Override baseline features with any explicit User inputs
    for key, value in user_inputs.items():
        if key in patient_raw:
            patient_raw[key] = value
            
    # 3. Create the 121-column DataFrame initialized to exactly 0.0 (which handles base categories automatically)
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
            
    # 6. Apply Standard Scaler
    # Ensuring the dataframe columns are in the exact same order expected by the scaler
    input_scaled = scaler.transform(input_df[expected_features])
    
    return input_scaled, input_df

# ==========================================
# EXAMPLE OF HOW TO INTEGRATE IN APP.PY
# ==========================================
if __name__ == "__main__":
    import joblib
    
    # Dummy mock for scaler test (In app.py, you would use joblib.load('scaler.joblib'))
    from sklearn.preprocessing import StandardScaler
    
    print("Testing functionality...")
    try:
        schema = load_schema()
        # We'll create a dummy scaler just to see if the transform shapes match
        scaler = StandardScaler()
        scaler.mean_ = np.zeros(121)
        scaler.scale_ = np.ones(121)
        
        # Example Streamlit bindings mapping dictionary!
        # Notice we fix 'Diabetic' -> 'HadDiabetes'
        mock_ui_inputs = {
            'AgeCategory': 'Age 50 to 54',
            'Sex': 'Male',
            'BMI': 28.5,
            'GeneralHealth': 'Fair',
            'SmokerStatus': 'Former smoker',
            'HadStroke': 'No',
            'HadDiabetes': 'No'   # <--- Notice the valid dataset column name!
        }
        
        input_scaled, raw_processed_df = prepare_patient_vector(mock_ui_inputs, schema, scaler)
        
        print("\n✅ Success! Generated vector shape:", input_scaled.shape)
        print("Set features count:", raw_processed_df.sum(axis=1).values[0], " (should equal exactly the number of categorical columns + scaled numerical sum)")
    except FileNotFoundError:
        print("Please run `python schema_builder.py` first to generate model_schema.json.")
