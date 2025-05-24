import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="SARS Forecasting Platform", layout="wide")

# Page style and layout
st.markdown("""
    <style>
        body {
            background-image: url('1B.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }
        html, body, [class*="css"] {
            background-color: rgba(0, 0, 0, 0.85);
            color: #f0f0f0;
        }
        .block-container {
            padding-top: 5rem;
            padding-left: 5%;
            padding-right: 5%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .section, .uploader-box {
            border: 1px solid #444;
            padding: 2rem;
            margin-bottom: 2rem;
            border-radius: 12px;
            background-color: rgba(30, 30, 30, 0.9);
            width: 80%;
            text-align: center;
        }
        h1, h2, h3, h4 {
            color: #00c3ff;
        }
        .stButton>button {
            background-color: #00c3ff;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Main heading
st.markdown("<div class='section'><h1>📊 SARS Forecasting Platform</h1></div>", unsafe_allow_html=True)

# Welcome section
st.markdown("""
<div class='section'>
    <h2>Welcome to the SARS Sales Forecasting Dashboard</h2>
    <p style="font-size: 16px;">
        Upload your last year's sales data below to generate actionable sales forecasts using ARIMAX.
        This platform helps you reduce stockouts, optimize inventory, and improve operational performance.
    </p>
</div>
""", unsafe_allow_html=True)

# File uploader
st.markdown("<div class='uploader-box'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📁 Upload your Last Year Sales File", type=["csv"])
st.markdown("</div>", unsafe_allow_html=True)

if not uploaded_file:
    st.warning("⚠️ Please upload a CSV file to proceed.")
    st.stop()

@st.cache_data
def load_data(file):
    df = pd.read_csv(file, parse_dates=['Date'])
    df.dropna(how='all', inplace=True)
    df = df.drop_duplicates()
    df = df[df['Sales'] > 0]
    df['Discount'] = df['Discount'].apply(lambda x: 1 if x == 'Yes' else 0)
    df = df.set_index('Date').asfreq('D').ffill()
    return df

try:
    df_check = pd.read_csv(uploaded_file)
    if 'Date' not in df_check.columns:
        st.error(f"❌ The uploaded file is missing a 'Date' column. Found columns: {df_check.columns.tolist()}")
        st.stop()
    raw_df = load_data(uploaded_file)
except Exception as e:
    st.error(f"❌ Failed to process file: {e}")
    st.stop()

# Sidebar filters
with st.sidebar.expander("🔎 **Filter Parameters**", expanded=True):
    st.markdown("""
    <div style='padding: 10px; background-color: #f0f2f6; border-radius: 10px;'>
        <strong>Choose filters and click Apply to update forecasts:</strong>
    </div>
    """, unsafe_allow_html=True)
    selected_region = st.selectbox("🌍 Select Region", sorted(raw_df['Region_Code'].unique()))
    selected_store_type = st.selectbox("🏪 Select Store Type", sorted(raw_df['Store_Type'].unique()))
    selected_location_type = st.selectbox("📍 Select Location Type", sorted(raw_df['Location_Type'].unique()))
    if st.button("✅ Apply Filters"):
        st.session_state.filters_applied = True

if 'filters_applied' not in st.session_state or not st.session_state.filters_applied:
    st.stop()

# Filtered and aggregated data
df = raw_df[(raw_df['Region_Code'] == selected_region) &
            (raw_df['Store_Type'] == selected_store_type) &
            (raw_df['Location_Type'] == selected_location_type)]

if df.empty:
    st.error("❌ No data available for selected filters. Please choose a different combination.")
    st.stop()

df = df.groupby('Date').agg({
    'Sales': 'sum',
    'Holiday': 'max',
    '#Order': 'sum',
    'Discount': 'sum',
    'Store_id': 'nunique'
}).asfreq('D').ffill()
df['Date'] = df.index

# Forecasting prep
one_year_ago = df.index.max() - pd.Timedelta(days=365)
latest_year = df.loc[df.index >= one_year_ago].copy()

if latest_year.empty:
    st.error("❌ Not enough data in the last 365 days to build a model.")
    st.stop()

latest_year['log_sales'] = np.log1p(latest_year['Sales'])
log_sales_series = latest_year['log_sales']
exog = latest_year[['Holiday', '#Order', 'Discount', 'Store_id']]

n = len(log_sales_series)
if n < 10:
    st.error("❌ Insufficient data points after filtering to train the model.")
    st.stop()

train_end = int(n * 0.8)
val_end = int(n * 0.9)
train_y, val_y = log_sales_series[:train_end], log_sales_series[train_end:val_end]
train_exog, val_exog = exog[:train_end], exog[train_end:val_end]

if len(train_y) == 0 or len(val_y) == 0:
    st.error("❌ Training or validation set is empty. Check the amount of available data.")
    st.stop()

model = ARIMA(train_y, order=(2, 1, 2), exog=train_exog)
model_fit = model.fit()

forecast_log = model_fit.forecast(steps=len(val_y), exog=val_exog)
forecast = np.expm1(forecast_log)
actual = np.expm1(val_y)
val_dates = latest_year.index[train_end:val_end]
rmse = np.sqrt(mean_squared_error(actual, forecast))

# Forecast DataFrame
forecast_df = pd.DataFrame({
    'Date': val_dates,
    'Actual Sales': actual.values,
    'Forecasted Sales': forecast.values
})
