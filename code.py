# ==============================
# UIDAI AADHAAR ANALYTICS DASHBOARD
# Enrolment + ML Forecast + Anomaly Detection
# ==============================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from prophet import Prophet

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="UIDAI Aadhaar Analytics",
    layout="wide"
)

# ------------------------------
# LOAD DATA
# ------------------------------
@st.cache_data
def load_data():
    enrol = pd.read_csv("enrolment_clean.csv")
    bio = pd.read_csv("biometric_clean.csv")

    enrol['date'] = pd.to_datetime(enrol['date'])
    bio['date'] = pd.to_datetime(bio['date'])

    enrol['year'] = enrol['date'].dt.year
    bio['year'] = bio['date'].dt.year

    return enrol, bio

enrolment_df, biometric_df = load_data()

# ------------------------------
# SIDEBAR FILTERS
# ------------------------------
st.sidebar.title("Filters")

year_selected = st.sidebar.selectbox(
    "Select Year",
    sorted(enrolment_df['year'].unique())
)

state_selected = st.sidebar.selectbox(
    "Select State",
    sorted(enrolment_df['state'].unique())
)

filtered_df = enrolment_df[
    (enrolment_df['year'] == year_selected) &
    (enrolment_df['state'] == state_selected)
]

# ------------------------------
# TITLE
# ------------------------------
st.title("📊 UIDAI Aadhaar Enrolment & Update Analytics Dashboard")
st.markdown("**Data Source:** UIDAI Open Government Data (data.gov.in)")

# ------------------------------
# KPI METRICS
# ------------------------------
st.subheader("📌 Key Metrics")

total_enrolments = filtered_df[['age_0_5','age_5_17','age_18_greater']].sum().sum()

c1, c2, c3 = st.columns(3)
c1.metric("Total Enrolments", f"{int(total_enrolments):,}")
c2.metric("Selected State", state_selected)
c3.metric("Selected Year", year_selected)

# ------------------------------
# YEAR-WISE ENROLMENT TREND
# ------------------------------
st.subheader("📈 Year-wise Aadhaar Enrolment Trend")

yearly = enrolment_df.groupby('year')[['age_0_5','age_5_17','age_18_greater']].sum()
yearly['total'] = yearly.sum(axis=1)

fig, ax = plt.subplots()
ax.plot(yearly.index, yearly['total'])
ax.set_xlabel("Year")
ax.set_ylabel("Total Enrolments")
st.pyplot(fig)

# ------------------------------
# STATE-WISE COMPARISON
# ------------------------------
st.subheader("🏙 Top States by Aadhaar Enrolment")

state_data = enrolment_df.groupby('state')[['age_0_5','age_5_17','age_18_greater']].sum()
state_data['total'] = state_data.sum(axis=1)
top_states = state_data.sort_values('total', ascending=False).head(10)

fig, ax = plt.subplots()
top_states['total'].plot(kind='bar', ax=ax)
ax.set_xlabel("State")
ax.set_ylabel("Enrolments")
st.pyplot(fig)

# ------------------------------
# BIOMETRIC UPDATE TREND
# ------------------------------
st.subheader("🧬 Biometric Update Trend")

bio_yearly = biometric_df.groupby('year')[['bio_age_5_17','bio_age_17_']].sum()
bio_yearly['total'] = bio_yearly.sum(axis=1)

fig, ax = plt.subplots()
ax.plot(bio_yearly.index, bio_yearly['total'])
ax.set_xlabel("Year")
ax.set_ylabel("Biometric Updates")
st.pyplot(fig)

# ------------------------------
# ANOMALY DETECTION (ISOLATION FOREST)
# ------------------------------
st.subheader("⚠️ Anomaly Detection in Enrolment Demand")

iso_df = yearly[['total']].copy()

model_iso = IsolationForest(
    contamination=0.1,
    random_state=42
)

iso_df['anomaly'] = model_iso.fit_predict(iso_df)

anomalies = iso_df[iso_df['anomaly'] == -1]

fig, ax = plt.subplots()
ax.plot(iso_df.index, iso_df['total'], label="Normal")
ax.scatter(anomalies.index, anomalies['total'], color='red', label="Anomaly")
ax.set_xlabel("Year")
ax.set_ylabel("Total Enrolments")
ax.legend()
st.pyplot(fig)

st.caption("Red points indicate abnormal enrolment surges requiring administrative attention.")

# ------------------------------
# ML FORECASTING (PROPHET)
# ------------------------------
st.subheader("🔮 Aadhaar Enrolment Forecast (ML Model)")

prophet_df = yearly.reset_index()[['year','total']]
prophet_df.columns = ['ds','y']
prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%Y')

model = Prophet(
    yearly_seasonality=False,
    daily_seasonality=False,
    weekly_seasonality=False
)

model.fit(prophet_df)

future = model.make_future_dataframe(periods=5, freq='Y')
forecast = model.predict(future)

fig1 = model.plot(forecast)
plt.xlabel("Year")
plt.ylabel("Total Enrolments")
st.pyplot(fig1)

st.caption("Forecasted Aadhaar enrolment demand for the next 5 years.")

# ------------------------------
# FORECAST COMPONENTS
# ------------------------------
st.subheader("📊 Forecast Trend Components")

fig2 = model.plot_components(forecast)
st.pyplot(fig2)

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.caption(
    "This dashboard is a supplementary visualization for UIDAI Data Hackathon 2026. "
    "Primary submission is the consolidated analytical PDF."
)