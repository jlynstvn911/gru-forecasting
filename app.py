import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
import os

st.set_page_config(page_title="GRU Forecasting App", layout="centered")
st.title("📈 GRU Stock Forecasting App")
st.write("Aplikasi ini memprediksi harga selanjutnya menggunakan model GRU.")

uploaded_file = st.file_uploader("Upload file CSV (harus ada kolom 'Close')", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data berhasil diupload.")
else:
    df = pd.read_csv("data.csv")
    st.info("🔍 Menampilkan data default (data.csv) karena tidak ada file yang diupload.")

st.subheader("Data Harga Saham")
st.write(df.tail())

df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA20'] = df['Close'].rolling(window=20).mean()

st.subheader("Visualisasi Harga dan Moving Average")
fig, ax = plt.subplots()
ax.plot(df['Close'], label='Close', color='blue')
ax.plot(df['MA10'], label='MA10', linestyle='--')
ax.plot(df['MA20'], label='MA20', linestyle='--')
ax.legend()
st.pyplot(fig)

scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(df[['Close']])

def create_dataset(data, time_step=10):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step])
        y.append(data[i+time_step])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_close, 10)

MODEL_PATH = "gru_model.h5"

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    st.success("✅ Model GRU berhasil dimuat.")
else:
    st.warning("⚠️ Model belum tersedia. Akan dilakukan training cepat.")
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = Sequential([
        GRU(64, return_sequences=False, input_shape=(10, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)
    model.save(MODEL_PATH)
    st.success("✅ Model sudah dilatih dan disimpan.")

last_10 = scaled_close[-10:].reshape(1, 10, 1)
pred_scaled = model.predict(last_10)
pred_price = scaler.inverse_transform(pred_scaled)

st.subheader("📊 Prediksi Harga Selanjutnya")
st.write(f"**Harga prediksi selanjutnya: {pred_price[0][0]:.2f}**")
