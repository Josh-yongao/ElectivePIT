#!/usr/bin/env python3
"""
Aggressive Model Training - Maximum F1 Score Focus
Uses aggressive class weighting, metric optimization, and threshold tuning
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report, brier_score_loss)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import joblib
import json
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("AGGRESSIVE MODEL TRAINING - MAXIMUM F1 FOCUS")
print("="*80)

# Load and prepare data
print("\n[1/5] Loading and preparing data...")
data_path = '/workspaces/ElectivePIT/Flood_Prediction_NCR_Philippines.csv'
df = pd.read_csv(data_path)

df_clean = df.copy()
df_clean = df_clean.drop_duplicates()

# Handle outliers
def handle_outliers(data, columns, multiplier=1.5):
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        data[col] = data[col].clip(lower_bound, upper_bound)
    return data

df_clean = handle_outliers(df_clean, ['Rainfall_mm', 'WaterLevel_m', 'SoilMoisture_pct'])

# Feature engineering
def add_lag_features(df, columns, lags=[1, 3, 7]):
    df = df.copy()
    for col in columns:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df.groupby('Location')[col].shift(lag)
    return df

def add_rolling_features(df, columns, windows=[3, 7]):
    df = df.copy()
    for col in columns:
        for window in windows:
            df[f'{col}_rolling_mean_{window}d'] = df.groupby('Location')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'{col}_rolling_sum_{window}d'] = df.groupby('Location')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).sum()
            )
    return df

df_features = add_lag_features(df_clean, ['Rainfall_mm', 'WaterLevel_m', 'SoilMoisture_pct'])
df_features = add_rolling_features(df_features, ['Rainfall_mm', 'WaterLevel_m'])
df_features = df_features.fillna(0)

# Temporal features
df_features['Date'] = pd.to_datetime(df_features['Date'], format='%d/%m/%Y')
df_features['Day_of_Year'] = df_features['Date'].dt.dayofyear
df_features['Month'] = df_features['Date'].dt.month
df_features['Quarter'] = df_features['Date'].dt.quarter
df_features['Week_of_Year'] = df_features['Date'].dt.isocalendar().week
df_features['Day_of_Week'] = df_features['Date'].dt.dayofweek
df_features['Is_Weekend'] = (df_features['Day_of_Week'] >= 5).astype(int)

location_dummies = pd.get_dummies(df_features['Location'], prefix='Location', drop_first=True)
df_features = pd.concat([df_features, location_dummies], axis=1)

X = df_features.drop(['Date', 'FloodOccurrence', 'Location'], axis=1)
y = df_features['FloodOccurrence']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=RANDOM_STATE, stratify=y_temp)

print(f"✓ Data prepared: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

# SMOTE with higher ratio to create synthetic positives
print("\n[2/5] Applying aggressive SMOTE...")
# Use SMOTE to create MORE synthetic flood samples
smote = SMOTE(sampling_strategy=0.8, random_state=RANDOM_STATE, k_neighbors=3)  # Create more positives
X_train_bal, y_train_bal = smote.fit_resample(X_train.values, y_train.values)
print(f"✓ After SMOTE: {np.bincount(y_train_bal)}")

# Scale
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train aggressive models with focus on recall
print("\n[3/5] Training aggressive models (heavy positive class weight)...")

# Random Forest - aggressive weights towards positives
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=30,
    min_samples_leaf=1,
    min_samples_split=2,
    max_features='sqrt',
    class_weight={0: 1, 1: 50},  # Strong positive class weight
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train_bal)

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.08,
    subsample=0.9,
    random_state=RANDOM_STATE
)
gb.fit(X_train_scaled, y_train_bal)

# XGBoost with scale_pos_weight
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=50,  # Heavy positive weight
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train_scaled, y_train_bal)

print("✓ Models trained")

# Optimize threshold for MAXIMUM F1
print("\n[4/5] Aggressive threshold optimization...")

y_val_proba_rf = rf.predict_proba(X_val_scaled)[:, 1]
y_val_proba_gb = gb.predict_proba(X_val_scaled)[:, 1]
y_val_proba_xgb = xgb_model.predict_proba(X_val_scaled)[:, 1]

# Ensemble probabilities
y_val_proba_ensemble = (y_val_proba_rf + y_val_proba_gb + y_val_proba_xgb) / 3

best_f1 = 0
best_threshold = 0.5
best_metrics = {}

# Try many thresholds including very low ones for aggressive recall
for threshold in np.arange(0.05, 0.95, 0.01):
    y_val_pred = (y_val_proba_ensemble >= threshold).astype(int)
    f1 = f1_score(y_val, y_val_pred, zero_division=0)
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_metrics = {
            'precision': precision_score(y_val, y_val_pred, zero_division=0),
            'recall': recall_score(y_val, y_val_pred, zero_division=0),
            'f1': f1
        }

print(f"✓ Best threshold: {best_threshold:.2f}")
print(f"  Val F1: {best_metrics['f1']:.4f}, Precision: {best_metrics['precision']:.4f}, Recall: {best_metrics['recall']:.4f}")

# Test set evaluation
print("\n[5/5] Evaluating on test set...")

y_test_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]
y_test_proba_gb = gb.predict_proba(X_test_scaled)[:, 1]
y_test_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
y_test_proba_ensemble = (y_test_proba_rf + y_test_proba_gb + y_test_proba_xgb) / 3

y_test_pred = (y_test_proba_ensemble >= best_threshold).astype(int)

test_metrics = {
    'accuracy': accuracy_score(y_test, y_test_pred),
    'precision': precision_score(y_test, y_test_pred, zero_division=0),
    'recall': recall_score(y_test, y_test_pred, zero_division=0),
    'f1': f1_score(y_test, y_test_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_test_proba_ensemble),
    'brier': brier_score_loss(y_test, y_test_proba_ensemble)
}

print("\n" + "="*80)
print("TEST SET PERFORMANCE - AGGRESSIVE TUNING")
print("="*80)
print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
print(f"Precision: {test_metrics['precision']:.4f}")
print(f"Recall:    {test_metrics['recall']:.4f}")
print(f"F1-Score:  {test_metrics['f1']:.4f}  ← TARGET: >= 0.90")
print(f"ROC-AUC:   {test_metrics['roc_auc']:.4f}")
print(f"Brier:     {test_metrics['brier']:.4f}")
print("="*80)

cm = confusion_matrix(y_test, y_test_pred)
print(f"\nConfusion Matrix:")
print(f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}")
print(f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['No Flood', 'Flood']))

# Save models
print("\n[SAVING] Model artifacts...")
os.makedirs('saved_models', exist_ok=True)

joblib.dump(rf, 'saved_models/random_forest_flood_prediction.pkl')
joblib.dump(gb, 'saved_models/gradient_boosting_model.pkl')
joblib.dump(xgb_model, 'saved_models/xgboost_model.pkl')
joblib.dump(scaler, 'saved_models/feature_scaler.pkl')

metadata = {
    'model_type': 'Aggressive Ensemble (RF + GB + XGBoost)',
    'n_features': len(X.columns),
    'feature_names': X.columns.tolist(),
    'best_threshold': float(best_threshold),
    'test_performance': {k: float(v) for k, v in test_metrics.items()},
    'training_date': datetime.now().isoformat(),
    'approach': 'SMOTE 0.8 + Aggressive Class Weight + Threshold Optimization'
}

with open('saved_models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

results = {
    'best_model': 'Aggressive Ensemble',
    'best_f1_score': float(test_metrics['f1']),
    'optimal_threshold': float(best_threshold),
    'test_metrics': {k: float(v) for k, v in test_metrics.items()},
    'confusion_matrix': cm.tolist()
}

with open('saved_models/training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✓ Models saved")

print("\n" + "="*80)
if test_metrics['f1'] >= 0.90:
    print("🎉 TARGET ACHIEVED: F1 >= 90%!")
else:
    print(f"Current F1: {test_metrics['f1']:.2%}")
print("="*80)
