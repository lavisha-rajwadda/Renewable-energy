import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


st.set_page_config(page_title="Solar Power & Peak Hour Dashboard", layout="wide")


st.title("Solar Power and Peak Hour Forecast Dashboard")
st.markdown(
    "Predict solar power, identify daily peak hours for the next week, compare renewable vs non-renewable energy usage, and get smart recommendations for load shifting and load addition based on your historical residential dataset."
)


# Data source
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload dataset CSV", type="csv")
default_path = "solar_residential_dataset.csv"


interval_minutes = st.sidebar.number_input(
    "Forecast interval (minutes)", min_value=5, max_value=60, value=60, step=5
)
forecast_days = st.sidebar.slider("Forecast days ahead", min_value=1, max_value=7, value=7)


target_col_power = "Solar Power Output (kW)"
target_col_demand = "Residential Load Demand (kW)"


@st.cache_data
def load_data():
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(default_path)
    # Normalize timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif "Timestamp" in df.columns:
        df.rename(columns={"Timestamp": "timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError("Dataset must contain a timestamp column named 'timestamp' or 'Timestamp'.")
    df.columns = [c.strip() for c in df.columns]
    return df


data = load_data()


st.subheader("Dataset Snapshot")
st.write(data.head())
st.write(f"Dataset shape: {data.shape}")


# Ensure necessary columns exist
required_cols = [
    "timestamp", "Solar Irradiance (W/m²)", "Ambient Temperature (°C)", "Humidity (%)",
    "Cloud Cover (%)", "Hour of Day", target_col_power, target_col_demand
]
missing = [c for c in required_cols if c not in data.columns]
if missing:
    st.error(f"Missing required columns in dataset: {missing}")
    st.stop()


if "Hour of Day" not in data.columns:
    data["Hour of Day"] = data["timestamp"].dt.hour
if "Day of Week" not in data.columns:
    data["Day of Week"] = data["timestamp"].dt.dayofweek


# === Monthly Energy Trend Visualization ===
st.markdown("## Monthly Consumption & Renewable vs Non-Renewable Energy")


# Group data by month
data['Month'] = data['timestamp'].dt.to_period('M').astype(str)
monthly_stats = (
    data.groupby('Month')[[target_col_power, target_col_demand]]
    .sum()
    .rename(columns={
        target_col_power: 'Total Solar Generated (kWh)',
        target_col_demand: 'Total Consumption (kWh)'
    })
    .reset_index()
)
interval_hours = interval_minutes / 60.0
monthly_stats['Total Solar Generated (kWh)'] *= interval_hours
monthly_stats['Total Consumption (kWh)'] *= interval_hours
monthly_stats['Non-Renewable (kWh)'] = (
    monthly_stats['Total Consumption (kWh)'] - monthly_stats['Total Solar Generated (kWh)']
).clip(lower=0)
monthly_stats['Renewable %'] = 100 * (
    monthly_stats['Total Solar Generated (kWh)'] / monthly_stats['Total Consumption (kWh)']
).fillna(0)


# Round for display
monthly_stats = monthly_stats.round(2)
st.write("Monthly summary table:", monthly_stats)


import plotly.graph_objs as go
fig = go.Figure()
fig.add_trace(go.Bar(
    x=monthly_stats['Month'],
    y=monthly_stats['Total Solar Generated (kWh)'],
    name='Solar (Renewable)'
))
fig.add_trace(go.Bar(
    x=monthly_stats['Month'],
    y=monthly_stats['Non-Renewable (kWh)'],
    name='Non-Renewable'
))
fig.update_layout(
    barmode='stack',
    yaxis_title='Energy (kWh)',
    title='Renewable vs Non-Renewable Energy by Month'
)
st.plotly_chart(fig, use_container_width=True)


st.line_chart(monthly_stats.set_index('Month')['Renewable %'])


# === Solar Power Prediction ===
st.markdown("## Solar Power Forecast for Next 7 Days")
X_power = data[[
    "Solar Irradiance (W/m²)", "Ambient Temperature (°C)", "Humidity (%)",
    "Cloud Cover (%)", "Hour of Day", "Day of Week"
]]
y_power = data[target_col_power]


X_train_p, X_val_p, y_train_p, y_val_p = train_test_split(X_power, y_power, test_size=0.2, random_state=42)
model_power = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model_power.fit(X_train_p, y_train_p)
y_pred_val_p = model_power.predict(X_val_p)
r2_p = r2_score(y_val_p, y_pred_val_p)
mae_p = mean_absolute_error(y_val_p, y_pred_val_p)


st.write(f"Solar Power Model R²: {r2_p:.4f}")
st.write(f"Solar Power MAE (kW): {mae_p:.4f}")


start_ts = data["timestamp"].max() + pd.Timedelta(minutes=interval_minutes)
n_steps = int((forecast_days * 24 * 60) / interval_minutes)
forecast_index = pd.date_range(start=start_ts, periods=n_steps, freq=f"{interval_minutes}T")


future_features = pd.DataFrame({
    "Solar Irradiance (W/m²)": [data["Solar Irradiance (W/m²)"].mean()] * n_steps,
    "Ambient Temperature (°C)": [data["Ambient Temperature (°C)"].mean()] * n_steps,
    "Humidity (%)": [data["Humidity (%)"].mean()] * n_steps,
    "Cloud Cover (%)": [data["Cloud Cover (%)"].mean()] * n_steps,
    "Hour of Day": forecast_index.hour,
    "Day of Week": forecast_index.dayofweek
})
future_pred_power = model_power.predict(future_features)
forecast_df = pd.DataFrame({"timestamp": forecast_index, "Predicted Solar Power (kW)": future_pred_power})
st.line_chart(forecast_df.set_index("timestamp")["Predicted Solar Power (kW)"])
st.write(forecast_df.head(24))


# ---- Recommend optimal load addition times based on forecasted solar generation ----
st.markdown("### Recommended Times for Load Addition")
solar_threshold = np.percentile(future_pred_power, 75)  # Top quartile: high generation
add_load_slots = forecast_df[forecast_df["Predicted Solar Power (kW)"] >= solar_threshold]
if not add_load_slots.empty:
    for _, row in add_load_slots.iterrows():
        hour = pd.to_datetime(row["timestamp"]).strftime('%Y-%m-%d %H:%M')
        st.success(
            f"**High solar generation expected at {hour} ({row['Predicted Solar Power (kW)']:.2f} kW)**\n"
            "- Recommended to run EV charging, washing machines, pumps, or schedule other high-consumption appliances to utilize solar power and minimize grid dependency."
        )
else:
    st.info("No sufficiently high solar generation times found for load addition in the forecasted window.")


# === Peak Hour Prediction ===
st.markdown("## Peak Consumption Hour Identification for Next 7 Days")


feature_cols_demand = [
    "Solar Irradiance (W/m²)", "Ambient Temperature (°C)", "Humidity (%)",
    "Cloud Cover (%)", "Hour of Day", "Day of Week"
]
X_demand = data[feature_cols_demand]
y_demand = data[target_col_demand]


X_train_d, X_val_d, y_train_d, y_val_d = train_test_split(X_demand, y_demand, test_size=0.2, random_state=42)
model_demand = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model_demand.fit(X_train_d, y_train_d)
y_pred_val_d = model_demand.predict(X_val_d)
r2_d = r2_score(y_val_d, y_pred_val_d)
mae_d = mean_absolute_error(y_val_d, y_pred_val_d)


st.write(f"Peak Demand Model R²: {r2_d:.4f}")
st.write(f"Peak Demand MAE (kW): {mae_d:.4f}")


future_pred_demand = model_demand.predict(future_features)
forecast_demand_df = pd.DataFrame({
    "timestamp": forecast_index,
    "Predicted Consumption (kW)": future_pred_demand,
    "Hour of Day": forecast_index.hour,
    "Day": forecast_index.date
})


# Find peak consumption hours for each day
peak_hours_list = []
for day, group in forecast_demand_df.groupby("Day"):
    peak_row = group.loc[group['Predicted Consumption (kW)'].idxmax()]
    peak_hours_list.append({
        "Day": day,
        "Peak Hour": int(peak_row["Hour of Day"]),
        "Predicted Consumption (kW)": round(peak_row["Predicted Consumption (kW)"], 2)
    })
peak_hours_df = pd.DataFrame(peak_hours_list)
st.write("Predicted Peak Consumption Hours for Next 7 Days")
st.dataframe(peak_hours_df, use_container_width=True)


# --- Recommendations during peak hours (load shifting) ---
st.markdown("### Load Shifting Recommendations During Predicted Peak Hours")
peak_rec_message = """
- Shift usage of energy-intensive appliances (e.g., EV charging, air conditioning, pumps, water heaters) **outside the peak hour** when possible.
- Pre-cool or pre-heat rooms before the peak hour.
- Run dishwashers, laundry, and water pumps in off-peak times.
- Turn off non-essential lights and devices during peak hour.
"""
for idx, row in peak_hours_df.iterrows():
    st.warning(
        f"**{row['Day']} | Peak Hour: {row['Peak Hour']}:00 | Predicted Load: {row['Predicted Consumption (kW)']} kW**\n\n"
        f"{peak_rec_message}"
    )
