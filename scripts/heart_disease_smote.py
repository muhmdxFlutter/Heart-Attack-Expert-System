# ================================================================
#  HEART DISEASE EXPERT SYSTEM — ELITE VERSION (2025 COMPLIANT)
#  Target: Accuracy > 92% (Frontiers AI 2025) & Recall > 80%
#  Architecture: Deep Dense Network with Swish & Focal Loss Logic
# ================================================================

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress annoying TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─────────────────────────────────────────────────────────────
# STEP 1: Data Loading & Integrity Check
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_NAME = os.path.join(BASE_DIR, 'data', 'heart_2022_no_nans.csv')
TARGET_COL = 'HadHeartAttack'

try:
    df = pd.read_csv(FILE_NAME)
    print(f'✅ Dataset Loaded: {len(df):,} records')
except Exception as e:
    print(f'❌ Error: Could not find {FILE_NAME}. Please ensure it is in the same directory.')
    exit()

# ─────────────────────────────────────────────────────────────
# STEP 2: Advanced Preprocessing (One-Hot Expansion)
# ─────────────────────────────────────────────────────────────
print('🛠️  Running High-Precision Preprocessing...')

# 1. Target mapping
target_map = {'Yes': 1, 'No': 0}
y_raw = df[TARGET_COL].map(target_map).fillna(0).astype('int32')

# 2. Process Features (One-Hot Encoding)
x_raw = df.drop(columns=[TARGET_COL])
cat_cols = x_raw.select_dtypes(exclude=[np.number]).columns.tolist()

# Convert categorical columns to independent one-hot columns
x_encoded = pd.get_dummies(x_raw, columns=cat_cols, drop_first=True, dtype=int)

X = x_encoded.astype('float32') # Keeping as DataFrame to maintain column names
Y = y_raw.values.astype('int32')

# 3. Train/Test Split
X_dev, X_test, Y_dev, Y_test = train_test_split(
    X, Y, test_size=0.15, random_state=SEED, stratify=Y
)

# 4. Compute balanced class weights for the medical data
weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y_dev), y=Y_dev)
class_weights = dict(enumerate(weights))

print(f'✅ Preprocessing Complete. Features Expanded to: {X.shape[1]}')

# ─────────────────────────────────────────────────────────────
# STEP 3: Elite Neural Network Architecture
# ─────────────────────────────────────────────────────────────
def build_expert_model(input_dim):
    model = Sequential([
        # Layer 1: Digest the expanded feature set
        Dense(256, activation='swish', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.4),
        
        # Layer 2: Extract complex patterns
        Dense(128, activation='swish'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Layer 3: Filter final signals
        Dense(64, activation='swish'),
        BatchNormalization(),
        
        # Output decision node
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=AdamW(learning_rate=0.001, weight_decay=0.01),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )
    return model

# ─────────────────────────────────────────────────────────────
# STEP 4: 10-Fold Validation (Evidence Generation)
# ─────────────────────────────────────────────────────────────
print('\n' + '='*40)
print('🚀 STARTING 10-FOLD EXPERT VALIDATION')
print('='*40)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev, Y_dev)):
    # Split data per fold using .iloc for DataFrame compatibility
    X_train_f, X_val_f = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
    Y_train_f, Y_val_f = Y_dev[train_idx], Y_dev[val_idx]
    
    # Feature Scaling
    scaler = preprocessing.StandardScaler()
    X_train_f = scaler.fit_transform(X_train_f)
    X_val_f = scaler.transform(X_val_f)
    
    # Balance data per fold using SMOTE to avoid data leakage
    sm = SMOTE(random_state=SEED)
    X_res, Y_res = sm.fit_resample(X_train_f, Y_train_f)
    
    # Train a quick evaluation model for each fold
    fold_model = build_expert_model(X.shape[1])
    fold_model.fit(
        X_res, Y_res, epochs=12, batch_size=1024, 
        class_weight=class_weights, verbose=0
    )
    
    # Evaluate with a sensitive threshold to improve recall
    val_probs = fold_model.predict(X_val_f, verbose=0)
    val_preds = (val_probs > 0.4).astype(int)
    
    acc = accuracy_score(Y_val_f, val_preds)
    cv_scores.append(acc)
    print(f'Fold {fold+1:02d} | Accuracy: {acc*100:.2f}%')

print(f'⭐ Average Cross-Validation Accuracy: {np.mean(cv_scores)*100:.2f}%')

# ─────────────────────────────────────────────────────────────
# STEP 5: Final Training (The High-Recall Model)
# ─────────────────────────────────────────────────────────────
print('\n🏆 Training Final Expert Model...')

# Standardize final training and test sets
sc = preprocessing.StandardScaler()
X_dev_scaled = sc.fit_transform(X_dev)
X_test_scaled = sc.transform(X_test)

# Balance the final training set
# We set minority class to 15% to avoid over-distorting the feature space
X_final_res, Y_final_res = SMOTE(sampling_strategy=0.15, random_state=SEED).fit_resample(X_dev_scaled, Y_dev)

expert_model = build_expert_model(X.shape[1])

# Smart training callbacks
callbacks = [
    EarlyStopping(monitor='val_recall', mode='max', patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

history = expert_model.fit(
    X_final_res, Y_final_res,
    validation_split=0.15,
    epochs=100,
    batch_size=512,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────────────────────────
# STEP 6: Final Evaluation
# ─────────────────────────────────────────────────────────────
# Adjusting the threshold to reach a balance between accuracy and recall
THRESHOLD = 0.9

Y_probs = expert_model.predict(X_test_scaled, verbose=0).flatten()
Y_preds = (Y_probs > THRESHOLD).astype(int)

print('\n' + '='*40)
print('📊  FINAL EXPERT SYSTEM PERFORMANCE')
print('='*40)
print(classification_report(Y_test, Y_preds))

# Save Confusion Matrix
cm = confusion_matrix(Y_test, Y_preds)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', cbar=False)
plt.title(f'Final Expert Matrix (Threshold: {THRESHOLD})')
plt.ylabel('Actual Heart Status')
plt.xlabel('Predicted Heart Status')
plt.savefig(os.path.join(BASE_DIR, 'assets', 'expert_system_final_cm.png'))

print('\n✅ Final Plot Saved: expert_system_final_cm.png')
print('🎯 System is ready for Deployment into Dashboard.')

# Save the model
expert_model.save(os.path.join(BASE_DIR, 'models', 'expert_model.h5'))

# Save the Scaler so it can be used identically in the Dashboard
import joblib
import json
joblib.dump(sc, os.path.join(BASE_DIR, 'models', 'scaler.joblib'))

print("✅ Model and Scaler saved successfully!")

# --- Automatic generation of model_schema.json ---
print("\n⚙️ Generating model_schema.json for App inference...")
schema = {
    'expected_features': list(x_encoded.columns),
    'numerical_features': x_raw.select_dtypes(include=[np.number]).columns.tolist(),
    'categorical_features': cat_cols,
    'baselines': {}
}

# Calculate baseline defaults (Median for numerical, Mode for categorical)
for col in x_raw.columns:
    if col in schema['numerical_features']:
        schema['baselines'][col] = float(x_raw[col].median())
    else:
        schema['baselines'][col] = str(x_raw[col].mode()[0])

with open(os.path.join(BASE_DIR, 'models', 'model_schema.json'), 'w') as f:
    json.dump(schema, f, indent=4)
print("✅ model_schema.json saved successfully! System is fully integrated.")