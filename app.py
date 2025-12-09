import streamlit as st
import joblib
import numpy as np

st.title("📈 BBCA Forecasting — ARIMA Model (Walk-Forward)")

# LOAD MODEL
model_obj = joblib.load("best_model_BCA.joblib")
model = model_obj["model"]
order = model_obj["order"]

st.subheader("🔍 Model Loaded dari .joblib")
st.json({
    "model_type": model_obj["model_type"],
    "order": order
})

# INPUT USER
st.write("Masukkan harga Close terakhir:")
harga_terakhir = st.number_input("Harga Close", value=0.0)

if st.button("PREDIKSI HARGA BESOK"):
    forecast = model.forecast(steps=1)
    prediksi = forecast[0]

    st.success(f"📊 Prediksi harga besok: **{prediksi:,.2f}**")
