import streamlit as st
import pandas as pd
import joblib
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

st.title("📈 ARIMA Forecasting App - BBCA")

@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_model_BCA.joblib")

        if not hasattr(model, "forecast"):
            st.warning("⚠️ Model belum di-fit, mencoba melakukan fit ulang...")
            if hasattr(model, "endog"):
                model = model.fit()
                st.success("✅ Model berhasil di-fit ulang.")
            else:
                st.error("❌ Model tidak memiliki data internal untuk di-fit.")
                return None

        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

model = load_model()

if model is not None:
    st.success("✅ Model ARIMA berhasil dimuat!")
    steps = st.number_input("Masukkan jumlah langkah (hari/minggu) ke depan untuk prediksi:", min_value=1, max_value=30, value=1)

    if st.button("🔮 Forecast Sekarang"):
        try:
            forecast = model.forecast(steps=steps)
            st.subheader("📊 Hasil Forecast")
            st.write(pd.DataFrame({
                "Step": np.arange(1, steps + 1),
                "Forecast": forecast
            }))
        except Exception as e:
            st.error(f"Terjadi kesalahan saat forecasting: {e}")
else:
    st.stop()
