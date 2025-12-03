# Quick Start Guide: Using the Flood Prediction Model

## Loading & Using the Trained Model

### 1. Import Required Libraries

```python
import joblib
import pandas as pd
import numpy as np
import json

# Load model artifacts
model_path = 'saved_models/random_forest_flood_prediction.pkl'
scaler_path = 'saved_models/feature_scaler.pkl'
metadata_path = 'saved_models/model_metadata.json'

rf_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
with open(metadata_path) as f:
    metadata = json.load(f)

print("✓ Model loaded successfully!")
print(f"Model Type: {metadata['model_type']}")
print(f"Number of Features: {metadata['n_features']}")
print(f"Test Set F1-Score: {metadata['test_performance']['f1_score']:.4f}")
```

### 2. Prepare Your Data

Your data should contain exactly these 30 features:

```python
required_features = metadata['feature_names']
print("Required features:")
for i, feat in enumerate(required_features, 1):
    print(f"  {i}. {feat}")
```

**Important**: Features must be in the exact order and include all engineered features:
- **Original**: Rainfall_mm, WaterLevel_m, SoilMoisture_pct, Elevation_m
- **Lag Features**: 1-day, 3-day, 7-day lags for rainfall, water level, soil moisture
- **Rolling Features**: 3-day and 7-day rolling means and sums
- **Temporal**: Day_of_Year, Month, Quarter, Week_of_Year, Day_of_Week, Is_Weekend
- **Location**: Location_Marikina, Location_Pasig, Location_Quezon City

### 3. Make Predictions

```python
# Example: Single day prediction
new_data = pd.DataFrame({
    # Your feature values here - must match order and count
})

# Make prediction
prediction = rf_model.predict(new_data)  # Returns 0 or 1
probability = rf_model.predict_proba(new_data)  # Returns [P(No Flood), P(Flood)]

print(f"Prediction: {'Flood' if prediction[0] == 1 else 'No Flood'}")
print(f"Flood Probability: {probability[0][1]:.2%}")

# Classify into risk levels
flood_prob = probability[0][1]
if flood_prob < 0.3:
    risk_level = "LOW"
elif flood_prob < 0.6:
    risk_level = "MEDIUM"
else:
    risk_level = "HIGH"

print(f"Risk Level: {risk_level}")
```

### 4. Batch Predictions (Multiple Days)

```python
# Predict for multiple days
new_data_batch = pd.DataFrame({
    # Multiple rows of feature data
})

predictions = rf_model.predict(new_data_batch)
probabilities = rf_model.predict_proba(new_data_batch)[:, 1]

results = pd.DataFrame({
    'Date': ['Day 1', 'Day 2', 'Day 3'],  # Your date column
    'Flood_Probability': probabilities,
    'Prediction': predictions,
    'Risk_Level': pd.cut(probabilities, 
                         bins=[0, 0.3, 0.6, 1.0],
                         labels=['Low', 'Medium', 'High'])
})

print(results)
```

## Feature Engineering from Raw Data

If you have only the raw weather data (Rainfall, WaterLevel, SoilMoisture, Elevation, Location, Date), 
you need to engineer the lag and rolling features:

```python
def prepare_raw_data(raw_df):
    """
    Prepare raw weather data with engineered features
    
    raw_df should contain:
    - Date (datetime)
    - Location (categorical)
    - Rainfall_mm (float)
    - WaterLevel_m (float)
    - SoilMoisture_pct (float)
    - Elevation_m (int)
    """
    
    df = raw_df.copy()
    
    # 1. Create lag features (grouped by location)
    for col in ['Rainfall_mm', 'WaterLevel_m', 'SoilMoisture_pct']:
        for lag in [1, 3, 7]:
            df[f'{col}_lag{lag}'] = df.groupby('Location')[col].shift(lag)
    
    # 2. Create rolling features (grouped by location)
    for col in ['Rainfall_mm', 'WaterLevel_m']:
        for window in [3, 7]:
            df[f'{col}_rolling_mean_{window}d'] = df.groupby('Location')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'{col}_rolling_sum_{window}d'] = df.groupby('Location')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).sum()
            )
    
    # 3. Create temporal features
    df['Day_of_Year'] = df['Date'].dt.dayofyear
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Week_of_Year'] = df['Date'].dt.isocalendar().week
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)
    
    # 4. One-hot encode location
    location_dummies = pd.get_dummies(df['Location'], prefix='Location', drop_first=True)
    df = pd.concat([df, location_dummies], axis=1)
    
    # 5. Fill NaN values from lag/rolling features
    df = df.fillna(0)
    
    # 6. Select only required columns in correct order
    df = df[metadata['feature_names']]
    
    return df

# Usage
raw_data = pd.read_csv('weather_data.csv')
raw_data['Date'] = pd.to_datetime(raw_data['Date'])
prepared_data = prepare_raw_data(raw_data)

# Make predictions
predictions = rf_model.predict(prepared_data)
probabilities = rf_model.predict_proba(prepared_data)[:, 1]
```

## Performance Summary

**Test Set Metrics**:
- Accuracy: 98.99%
- Precision: 65.51% (65 out of 100 flood predictions are correct)
- Recall: 95.00% (95% of actual floods are detected)
- F1-Score: 77.55%
- ROC-AUC: 0.9972

**Interpretation**:
- The model is very good at identifying true negatives (no-flood days)
- High recall means it catches most flood events (only 1 missed in test set)
- Lower precision means some false alarms (10 in test set)
- This is good for disaster preparedness (better safe than sorry)

## Common Issues & Solutions

### Issue 1: Feature Count Mismatch
**Error**: `ValueError: X has 25 features but model expects 30`

**Solution**: Ensure all engineered features are created. Check that:
1. All 30 features are present in your data
2. Features are in the correct order (see metadata['feature_names'])
3. All lag and rolling statistics are computed

### Issue 2: NaN Values
**Error**: `ValueError: Input contains NaN, infinity or a value too large`

**Solution**: Fill NaN values appropriately:
```python
# For lag/rolling features created at the beginning of time series
df = df.fillna(0)  # Or df = df.fillna(method='bfill')

# For missing weather data
df['Rainfall_mm'] = df['Rainfall_mm'].fillna(df['Rainfall_mm'].median())
```

### Issue 3: Location Not One-Hot Encoded
**Error**: Model predicts but accuracy is poor

**Solution**: Ensure location encoding uses the exact categories:
```python
# These three columns are required:
# - Location_Marikina (1 if Marikina, 0 otherwise)
# - Location_Pasig (1 if Pasig, 0 otherwise)  
# - Location_Quezon City (1 if Quezon City, 0 otherwise)
# - (Manila is the reference category, not included)

df['Location_Marikina'] = (df['Location'] == 'Marikina').astype(int)
df['Location_Pasig'] = (df['Location'] == 'Pasig').astype(int)
df['Location_Quezon City'] = (df['Location'] == 'Quezon City').astype(int)
```

## Operational Deployment

### Daily Batch Prediction Pipeline

```python
import schedule
import time
from datetime import datetime, timedelta

def predict_todays_flood_risk():
    """Run daily at 6 AM to predict today's flood risk"""
    
    # 1. Fetch today's weather data from PAGASA/MMDA APIs
    today = datetime.now().date()
    weather_data = fetch_weather_data(today)  # Your data source
    
    # 2. Prepare data with engineered features
    prepared_data = prepare_raw_data(weather_data)
    
    # 3. Make predictions
    probabilities = rf_model.predict_proba(prepared_data)[:, 1]
    
    # 4. Classify risk levels
    risk_levels = pd.cut(probabilities, 
                         bins=[0, 0.3, 0.6, 1.0],
                         labels=['Low', 'Medium', 'High'])
    
    # 5. Send alerts if needed
    for location, prob, risk in zip(weather_data['Location'], probabilities, risk_levels):
        if risk == 'High':
            send_alert(location, f"HIGH flood risk ({prob:.0%} probability)")
        elif risk == 'Medium':
            log_advisory(location, f"MEDIUM flood risk ({prob:.0%} probability)")
    
    return weather_data.assign(Probability=probabilities, RiskLevel=risk_levels)

# Schedule to run daily at 6 AM
schedule.every().day.at("06:00").do(predict_todays_flood_risk)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Alert Thresholds (Recommended)

- **HIGH (>60% probability)**: Issue evacuation alerts, activate emergency centers
- **MEDIUM (30-60% probability)**: Alert traffic management, increase drainage operations
- **LOW (<30% probability)**: Routine operations, standard advisories

## Integration Example: MMDA Alert System

```python
import requests
import json

def send_mmda_alert(location, flood_probability, risk_level):
    """Send flood alert to MMDA system"""
    
    payload = {
        'location': location,
        'flood_probability': float(flood_probability),
        'risk_level': risk_level,
        'timestamp': datetime.now().isoformat(),
        'model_version': 'RandomForest_v1.0',
        'model_accuracy': 0.9899
    }
    
    response = requests.post(
        'https://mmda.api.endpoint/flood-alerts',
        json=payload,
        headers={'Authorization': 'Bearer YOUR_API_KEY'}
    )
    
    return response.status_code == 200

# Usage
for idx, row in results.iterrows():
    success = send_mmda_alert(
        location=row['Location'],
        flood_probability=row['Flood_Probability'],
        risk_level=row['Risk_Level']
    )
    print(f"Alert sent to {row['Location']}: {success}")
```

## Model Maintenance

### Monthly Tasks
- Collect new flood/no-flood labels for validation
- Monitor model performance metrics
- Review false positives and false negatives

### Quarterly Tasks
- Retrain model with new data
- Validate on recent test set
- Update calibration if needed

### Annual Tasks
- Full hyperparameter tuning
- Comparison with alternative models (XGBoost, etc.)
- Comprehensive stakeholder review

---

**For more information**, refer to:
- `PROJECT_COMPLETION_SUMMARY.md` - Detailed project documentation
- `Flood_Prediction_Model.ipynb` - Full technical notebook
- `saved_models/model_metadata.json` - Model metadata and hyperparameters
