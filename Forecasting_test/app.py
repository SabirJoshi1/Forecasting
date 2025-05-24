import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
st.set_page_config(page_title="SARS Forecasting Platform", layout="wide")

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
        .section, .uploader-box, .chart-box {
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
            text-align: center;
        }
        p, label, div[data-testid="stMarkdownContainer"] {
            text-align: center;
        }
        .stButton>button {
            background-color: #00c3ff;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Centered Main Heading
st.markdown("<h1>📊 SARS Forecasting Platform</h1>", unsafe_allow_html=True)

# Welcome & Instructions
st.markdown("""
<div class='section'>
    <h2>Welcome to the SARS Sales Forecasting Dashboard</h2>
    <p style="font-size: 16px;">
        Upload your last year's sales data below to generate actionable sales forecasts using ARIMAX.
        This platform helps you reduce stockouts, optimize inventory, and improve operational performance.
    </p>
</div>
""", unsafe_allow_html=True)

# Centered File Uploader in a box
st.markdown("<div class='uploader-box'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📁 Upload your Last Year Sales File", type=["csv"])
st.markdown("</div>", unsafe_allow_html=True)

if not uploaded_file:
    st.warning("⚠️ Please upload a CSV file to proceed.")
    st.stop()

@st.cache_data
def load_data(file):
    df = pd.read_csv(file, parse_dates=['Date'])
    df['Discount'] = df['Discount'].apply(lambda x: 1 if x == 'Yes' else 0)
    return df

raw_df = load_data(uploaded_file)

if 'filters_applied' not in st.session_state:
    st.session_state.filters_applied = False

with st.sidebar.expander("🎛️ Choose Your Forecasting Scenario", expanded=True):
    selected_region = st.selectbox("🌍 Select Region", sorted(raw_df['Region_Code'].unique()))
    selected_store_type = st.selectbox("🏪 Select Store Type", sorted(raw_df['Store_Type'].unique()))
    selected_location_type = st.selectbox("📍 Select Location Type", sorted(raw_df['Location_Type'].unique()))
    if st.button("✅ Apply Filters"):
        st.session_state.filters_applied = True

if not st.session_state.filters_applied:
    st.stop()

df = raw_df[(raw_df['Region_Code'] == selected_region) &
            (raw_df['Store_Type'] == selected_store_type) &
            (raw_df['Location_Type'] == selected_location_type)]

df = df.groupby('Date').agg({
    'Sales': 'sum',
    'Holiday': 'max',
    '#Order': 'sum',
    'Discount': 'sum',
    'Store_id': 'nunique'
}).asfreq('D').fillna(method='ffill')
df['Date'] = df.index

latest_year = df.last('365D')
latest_year['log_sales'] = np.log1p(latest_year['Sales'])

log_sales_series = latest_year['log_sales']
exog = latest_year[['Holiday', '#Order', 'Discount', 'Store_id']]
n = len(log_sales_series)
train_end = int(n * 0.8)
val_end = int(n * 0.9)
train_y, val_y, _ = log_sales_series[:train_end], log_sales_series[train_end:val_end], log_sales_series[val_end:]
train_exog, val_exog, _ = exog[:train_end], exog[train_end:val_end], exog[val_end:]

model = ARIMA(train_y, order=(2, 1, 2), exog=train_exog)
model_fit = model.fit()

forecast_log = model_fit.forecast(steps=len(val_y), exog=val_exog)
forecast = np.expm1(forecast_log)
actual = np.expm1(val_y)
val_dates = latest_year.index[train_end:val_end]
rmse = np.sqrt(mean_squared_error(actual, forecast))

# Tabs continue from here with consistent styling...
