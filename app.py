import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

st.set_page_config(page_title="Metro Manila Flood Risk — Demo", layout="wide")

@st.cache_resource
def load_artifacts():
    # Load the best trained ensemble model
    try:
        model = joblib.load('saved_models/ensemble_flood_prediction.pkl')
    except FileNotFoundError:
        # Fallback to random forest if ensemble not available
        model = joblib.load('saved_models/random_forest_flood_prediction.pkl')
    
    try:
        calibrated = joblib.load('saved_models/calibrated_random_forest.pkl')
    except Exception:
        calibrated = None
    
    scaler = joblib.load('saved_models/feature_scaler.pkl')
    
    with open('saved_models/model_metadata.json') as f:
        metadata = json.load(f)
    
    return model, calibrated, scaler, metadata

model, calibrated_model, scaler, metadata = load_artifacts()
FEATURE_NAMES = metadata['feature_names']

# Small visual theming
st.markdown(
    """
    <style>
    .stApp { background-color: #000000; }
    .big-title {font-size:28px; font-weight:600; color:#0b5fff;}
    .subtle {color:#aaa; margin-top: -8px; margin-bottom:12px}
    .card {padding:12px; border-radius:8px; background:linear-gradient(90deg,#111111,#1a1a2e); box-shadow: 0 1px 3px rgba(13,26,62,0.3);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">Metro Manila Daily Flood Risk</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">A machine learning model predicting daily flood risk (Flood=1 / No Flood=0)</div>', unsafe_allow_html=True)

# Model info stored but not displayed in header
model_name = metadata.get('model_type', 'RandomForest')
train_date = metadata.get('training_date', metadata.get('training_timestamp', 'unknown'))
perf = metadata.get('test_performance', {}) or {}
f1 = perf.get('f1') or perf.get('f1_score') or perf.get('f1_score_test')
roc = perf.get('roc_auc') or perf.get('roc_auc_score')
brier = perf.get('brier_score') or perf.get('brier')

# Sidebar
st.sidebar.header('Options')
mode = st.sidebar.radio('Select mode', ['Manual Input', 'Demo (use included dataset)', 'Upload raw CSV (engineer features)', 'Upload prepared features CSV'])
threshold = st.sidebar.slider('Probability threshold (for Flood decision)', 0.0, 1.0, 0.5, 0.01)

# Helper functions

def prepare_raw_data(raw_df):
    df = raw_df.copy()
    # Expect Date, Location, Rainfall_mm, WaterLevel_m, SoilMoisture_pct, Elevation_m
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    # Sort
    df = df.sort_values(['Location', 'Date']).reset_index(drop=True)
    # Create lag features
    for col in ['Rainfall_mm', 'WaterLevel_m', 'SoilMoisture_pct']:
        for lag in [1, 3, 7]:
            df[f'{col}_lag{lag}'] = df.groupby('Location')[col].shift(lag)
    # Rolling features
    for col in ['Rainfall_mm', 'WaterLevel_m']:
        for window in [3, 7]:
            df[f'{col}_rolling_mean_{window}d'] = df.groupby('Location')[col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            df[f'{col}_rolling_sum_{window}d'] = df.groupby('Location')[col].transform(lambda x: x.rolling(window=window, min_periods=1).sum())
    # Temporal
    df['Day_of_Year'] = df['Date'].dt.dayofyear
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Week_of_Year'] = df['Date'].dt.isocalendar().week
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)
    # One-hot locations (matches the notebook's approach: drop_first=True)
    loc_dummies = pd.get_dummies(df['Location'], prefix='Location', drop_first=True)
    df = pd.concat([df, loc_dummies], axis=1)
    # Fill NaNs introduced by shifts
    df = df.fillna(0)
    # Ensure required features exist; if missing add zeros
    for feat in FEATURE_NAMES:
        if feat not in df.columns:
            df[feat] = 0
    # Keep only the last available record per date-location pair for prediction (i.e., rows retain as-is)
    prepared = df.copy()
    prepared = prepared.reset_index(drop=True)
    return prepared


def risk_color(risk_level):
    """Return color hex for risk level."""
    if risk_level == 'High':
        return '#ff6b6b'
    elif risk_level == 'Medium':
        return '#ffd43b'
    else:
        return '#51cf66'


def predict_on_prepared(X_df):
    X = X_df[FEATURE_NAMES].copy()
    # Scale using saved scaler if exists
    try:
        X_scaled = scaler.transform(X)
    except Exception:
        X_scaled = X.values
    # Use calibrated model if available for probabilities
    if calibrated_model is not None:
        probs = calibrated_model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= threshold).astype(int)
    else:
        probs = model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= threshold).astype(int)
    result = X_df.copy()
    result['Flood_Probability'] = probs
    result['Prediction'] = preds
    result['Risk_Level'] = pd.cut(probs, bins=[0, 0.3, 0.6, 1.0], labels=['Low', 'Medium', 'High'])
    return result


# Mode behaviour
if mode == 'Manual Input':
    st.subheader('Manual input for a single day and station')
    with st.form('manual_input_form'):
        c1, c2 = st.columns([2, 3])
        with c1:
            input_date = st.date_input('Date', value=datetime.today())
            input_location = st.selectbox('Location', options=['Quezon City','Marikina','Manila','Pasig'])
        with c2:
            rainfall = st.number_input('Rainfall (mm)', min_value=0.0, value=5.0, step=0.1)
            water_level = st.number_input('Water Level (m)', min_value=0.0, value=1.0, step=0.1)
            soil_moisture = st.number_input('Soil Moisture (%)', min_value=0.0, max_value=100.0, value=15.0, step=0.1)
            elevation = st.number_input('Elevation (m)', min_value=0.0, value=10.0, step=1.0)
        submitted = st.form_submit_button('Predict')
    if submitted:
        # Build a single-row raw DataFrame
        raw = pd.DataFrame([{ 'Date': input_date.strftime('%d/%m/%Y'),
                              'Location': input_location,
                              'Rainfall_mm': rainfall,
                              'WaterLevel_m': water_level,
                              'SoilMoisture_pct': soil_moisture,
                              'Elevation_m': elevation }])
        prepared = prepare_raw_data(raw)
        # prepared may contain multiple rows if same station has history; pick last row for prediction
        pred_row = prepared.iloc[[-1]]
        results = predict_on_prepared(pred_row)
        st.success('Prediction completed')
        st.table(results[['Date','Location','Rainfall_mm','WaterLevel_m','SoilMoisture_pct','Flood_Probability','Risk_Level','Prediction']])
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(label='Download prediction CSV', data=csv, file_name='manual_prediction.csv', mime='text/csv')

elif mode == 'Demo (use included dataset)':
    st.subheader('🌊 Demo using bundled dataset')
    st.markdown('Loads **Flood_Prediction_NCR_Philippines.csv**, engineers features, and predicts for the latest date per station.')
    if st.button('▶ Run demo predictions', use_container_width=True):
        raw = pd.read_csv('Flood_Prediction_NCR_Philippines.csv')
        prepared = prepare_raw_data(raw)
        # We'll show latest date per location
        latest = prepared.groupby(['Location']).apply(lambda g: g.loc[g['Date'].idxmax()]).reset_index(drop=True)
        results = predict_on_prepared(latest)
        st.success('✓ Predictions completed — showing latest date per station')
        
        with st.expander('📊 View full prediction table', expanded=True):
            st.dataframe(results[['Date','Location','Rainfall_mm','WaterLevel_m','SoilMoisture_pct','Flood_Probability','Risk_Level','Prediction']], use_container_width=True)
        
        flood_count = (results['Prediction'] == 1).sum()
        no_flood_count = (results['Prediction'] == 0).sum()
        with st.expander('📈 Summary statistics'):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.metric('Flood predictions', flood_count)
            with sc2:
                st.metric('No Flood predictions', no_flood_count)
        
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(label='⬇ Download predictions CSV', data=csv, file_name='demo_predictions.csv', mime='text/csv', use_container_width=True)

elif mode == 'Upload raw CSV (engineer features)':
    st.subheader('📤 Upload raw CSV (engineer features)')
    st.markdown('Columns expected: **Date, Location, Rainfall_mm, WaterLevel_m, SoilMoisture_pct, Elevation_m**')
    uploaded = st.file_uploader('Choose CSV file', type=['csv'])
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        with st.expander('Raw data preview', expanded=False):
            st.dataframe(raw.head(), use_container_width=True)
        
        prepared = prepare_raw_data(raw)
        with st.expander('Engineered features preview (first 10 rows)', expanded=False):
            st.dataframe(prepared[FEATURE_NAMES].head(10), use_container_width=True)
        
        if st.button('▶ Predict on uploaded data', use_container_width=True):
            results = predict_on_prepared(prepared)
            st.success(f'✓ Predictions completed for {len(results)} rows')
            with st.expander('📊 View predictions', expanded=True):
                st.dataframe(results[['Date','Location','Flood_Probability','Risk_Level','Prediction']].head(50), use_container_width=True)
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(label='⬇ Download predictions CSV', data=csv, file_name='uploaded_predictions.csv', mime='text/csv', use_container_width=True)

else:
    st.subheader('📤 Upload prepared features CSV')
    st.markdown('Upload a CSV that **already contains** engineered features in the required order.')
    uploaded = st.file_uploader('Choose prepared features CSV', type=['csv'])
    if uploaded is not None:
        dfp = pd.read_csv(uploaded)
        missing = [f for f in FEATURE_NAMES if f not in dfp.columns]
        if missing:
            st.error(f'❌ Missing required feature columns: {missing}')
        else:
            with st.expander('Prepared data preview', expanded=False):
                st.dataframe(dfp[FEATURE_NAMES].head(), use_container_width=True)
            
            if st.button('▶ Predict on prepared features', use_container_width=True):
                results = predict_on_prepared(dfp)
                st.success(f'✓ Predictions completed for {len(results)} rows')
                with st.expander('📊 View predictions', expanded=True):
                    st.dataframe(results[['Date','Location','Flood_Probability','Risk_Level','Prediction']].head(50), use_container_width=True)
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(label='⬇ Download predictions CSV', data=csv, file_name='prepared_predictions.csv', mime='text/csv', use_container_width=True)

# Sidebar: model info and top features
st.sidebar.markdown('---')
st.sidebar.header('🤖 Model Info')
st.sidebar.write(f"**Type:** {metadata.get('model_type')}")
st.sidebar.write(f"**Features:** {metadata.get('n_features')}")
if perf:
    if f1 is not None:
        st.sidebar.write(f"**Test F1:** {float(f1):.3f}")
    if roc is not None:
        st.sidebar.write(f"**Test ROC-AUC:** {float(roc):.3f}")

if st.sidebar.checkbox('📊 Show top feature importances'):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as _pd
        fi = model.feature_importances_
        idx = np.argsort(fi)[-15:]
        top_feats = [FEATURE_NAMES[i] for i in idx[::-1]]
        top_vals = fi[idx[::-1]]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.barh(range(len(top_feats)), top_vals[::-1], color='#2ecc71')
        ax.set_yticks(range(len(top_feats)))
        ax.set_yticklabels(top_feats[::-1])
        ax.set_xlabel('Importance')
        ax.set_title('Top 15 Feature Importances')
        st.sidebar.pyplot(fig)
    except Exception as e:
        st.sidebar.error(f'Could not display importances: {e}')

st.sidebar.markdown('---')
if st.sidebar.checkbox('📋 Show full metadata'):
    st.sidebar.json(metadata)
