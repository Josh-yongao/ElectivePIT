#!/usr/bin/env python3
"""
Improved Random Forest Model Training for Flood Prediction
Techniques: SMOTE, Threshold Tuning, Ensemble Methods, Feature Selection
Goal: Achieve F1-Score >= 90%
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report, brier_score_loss)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import joblib
import json

# Imbalanced learning
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 80)
print("IMPROVED MODEL TRAINING PIPELINE - TARGET F1 >= 90%")
print("=" * 80)

# ============================================================================
# 1. DATA LOADING AND PREPARATION
# ============================================================================
print("\n[1/6] Loading and preparing data...")

data_path = '/workspaces/ElectivePIT/Flood_Prediction_NCR_Philippines.csv'
df = pd.read_csv(data_path)

# Data cleaning
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

# Location encoding
location_dummies = pd.get_dummies(df_features['Location'], prefix='Location', drop_first=True)
df_features = pd.concat([df_features, location_dummies], axis=1)

# Prepare X and y
X = df_features.drop(['Date', 'FloodOccurrence', 'Location'], axis=1)
y = df_features['FloodOccurrence']

# Train/Val/Test split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=RANDOM_STATE, stratify=y_temp)

print(f"✓ Data prepared: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
print(f"  Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

# ============================================================================
# 2. SMOTE + UNDERSAMPLING FOR BALANCED TRAINING
# ============================================================================
print("\n[2/6] Applying SMOTE + Undersampling...")

X_train_balanced, y_train_balanced = SMOTE(random_state=RANDOM_STATE, k_neighbors=3).fit_resample(X_train.values, y_train.values)
print(f"✓ After SMOTE: {np.bincount(y_train_balanced)}")

# ============================================================================
# 3. FEATURE SCALING
# ============================================================================
print("\n[3/6] Scaling features...")

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Features scaled using RobustScaler")

# ============================================================================
# 4. ADVANCED MODEL TRAINING - ENSEMBLE APPROACH
# ============================================================================
print("\n[4/6] Training ensemble models...")

# Random Forest with optimized hyperparameters
print("  → Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_leaf=2,
    min_samples_split=3,
    max_features='log2',
    class_weight='balanced_subsample',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train_balanced)

# Gradient Boosting
print("  → Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    random_state=RANDOM_STATE
)
gb_model.fit(X_train_scaled, y_train_balanced)

# XGBoost
print("  → Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=250,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train_balanced == 0).sum() / (y_train_balanced == 1).sum(),
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train_scaled, y_train_balanced)

# Voting Ensemble
print("  → Creating Voting Classifier...")
voting_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('gb', gb_model),
        ('xgb', xgb_model)
    ],
    voting='soft'
)
voting_model.fit(X_train_scaled, y_train_balanced)

print("✓ All models trained successfully")

# ============================================================================
# 5. THRESHOLD OPTIMIZATION
# ============================================================================
print("\n[5/6] Optimizing probability threshold for F1 score...")

# Get predictions on validation set
y_val_proba_ensemble = voting_model.predict_proba(X_val_scaled)[:, 1]

best_f1_val = 0
best_threshold = 0.5
threshold_results = []

for threshold in np.arange(0.1, 0.95, 0.02):
    y_val_pred = (y_val_proba_ensemble >= threshold).astype(int)
    f1 = f1_score(y_val, y_val_pred, zero_division=0)
    precision = precision_score(y_val, y_val_pred, zero_division=0)
    recall = recall_score(y_val, y_val_pred, zero_division=0)
    threshold_results.append({'threshold': threshold, 'f1': f1, 'precision': precision, 'recall': recall})
    
    if f1 > best_f1_val:
        best_f1_val = f1
        best_threshold = threshold

print(f"✓ Optimal threshold: {best_threshold:.2f} (Validation F1: {best_f1_val:.4f})")

# ============================================================================
# 6. TEST SET EVALUATION
# ============================================================================
print("\n[6/6] Evaluating on test set with optimal threshold...")

y_test_proba_ensemble = voting_model.predict_proba(X_test_scaled)[:, 1]
y_test_pred_optimal = (y_test_proba_ensemble >= best_threshold).astype(int)

test_metrics = {
    'accuracy': accuracy_score(y_test, y_test_pred_optimal),
    'precision': precision_score(y_test, y_test_pred_optimal, zero_division=0),
    'recall': recall_score(y_test, y_test_pred_optimal, zero_division=0),
    'f1': f1_score(y_test, y_test_pred_optimal, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_test_proba_ensemble),
    'brier': brier_score_loss(y_test, y_test_proba_ensemble)
}

print("\n" + "=" * 80)
print("TEST SET PERFORMANCE (ENSEMBLE + THRESHOLD OPTIMIZATION)")
print("=" * 80)
print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
print(f"Precision: {test_metrics['precision']:.4f}")
print(f"Recall:    {test_metrics['recall']:.4f}")
print(f"F1-Score:  {test_metrics['f1']:.4f}  ← TARGET: >= 0.90")
print(f"ROC-AUC:   {test_metrics['roc_auc']:.4f}")
print(f"Brier:     {test_metrics['brier']:.4f}")
print("=" * 80)

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred_optimal)
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {cm[0, 0]}")
print(f"  False Positives: {cm[0, 1]}")
print(f"  False Negatives: {cm[1, 0]}")
print(f"  True Positives:  {cm[1, 1]}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred_optimal, target_names=['No Flood', 'Flood']))

# ============================================================================
# 7. SAVE IMPROVED MODEL
# ============================================================================
print("\n[SAVING] Persisting improved model artifacts...")

os.makedirs('saved_models', exist_ok=True)

# Save ensemble model
joblib.dump(voting_model, 'saved_models/ensemble_flood_prediction.pkl')
print("✓ Saved: ensemble_flood_prediction.pkl")

# Save individual models
joblib.dump(rf_model, 'saved_models/random_forest_improved.pkl')
joblib.dump(gb_model, 'saved_models/gradient_boosting_model.pkl')
joblib.dump(xgb_model, 'saved_models/xgboost_model.pkl')
print("✓ Saved: individual model components")

# Save scaler
joblib.dump(scaler, 'saved_models/feature_scaler.pkl')
print("✓ Saved: feature_scaler.pkl")

# Save metadata
metadata = {
    'model_type': 'VotingClassifier (RF + GB + XGBoost)',
    'n_features': len(X.columns),
    'feature_names': X.columns.tolist(),
    'best_threshold': float(best_threshold),
    'test_performance': {k: float(v) for k, v in test_metrics.items()},
    'training_date': datetime.now().isoformat(),
    'training_approach': 'SMOTE + Feature Engineering + Ensemble + Threshold Optimization'
}

with open('saved_models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✓ Saved: model_metadata.json")

# Update training results
results = {
    'best_model': 'Voting Ensemble',
    'best_f1_score': float(test_metrics['f1']),
    'optimal_threshold': float(best_threshold),
    'test_metrics': {k: float(v) for k, v in test_metrics.items()},
    'confusion_matrix': cm.tolist(),
    'threshold_optimization_results': threshold_results
}

with open('saved_models/training_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✓ Saved: training_results.json")

print("\n" + "=" * 80)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 80)
print(f"\nFinal F1-Score: {test_metrics['f1']:.4f}")
if test_metrics['f1'] >= 0.90:
    print("🎉 TARGET ACHIEVED: F1-Score >= 90%!")
else:
    print(f"⚠ F1-Score below 90%. Current: {test_metrics['f1']:.2%}")
print("=" * 80)
