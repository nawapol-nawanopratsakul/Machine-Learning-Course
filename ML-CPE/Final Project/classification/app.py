import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Car Price Prediction (1D CNN)", page_icon="🚗", layout="centered"
)


@st.cache_resource
def load_artifacts():
    output_dir = "output"
    model = load_model(os.path.join(output_dir, "car_cnn_model.keras"))
    scaler = joblib.load(os.path.join(output_dir, "scaler.pkl"))
    label_encoders = joblib.load(os.path.join(output_dir, "label_encoders.pkl"))
    return model, scaler, label_encoders


model, scaler, label_encoders = load_artifacts()

st.title("🚗 ระบบทำนายราคารถยนต์มือสอง (1D CNN)")
st.write("กรอกข้อมูลรถยนต์ของคุณเพื่อประเมินราคาขายจริงเป็นบาท")

with st.form("car_form"):
    year = st.number_input(
        "ปีที่ผลิต (Year)", min_value=1990, max_value=2026, value=2018
    )
    km_driven = st.number_input(
        "ระยะทางที่ขับมา (Km Driven)", min_value=0, value=50000
    )
    fuel = st.selectbox(
        "ประเภทเชื้อเพลิง (Fuel)", ["Petrol", "Diesel", "CNG", "LPG"]
    )
    seller_type = st.selectbox(
        "ประเภทผู้ขาย (Seller Type)",
        ["Individual", "Dealer", "Trustmark Dealer"],
    )
    transmission = st.selectbox(
        "ระบบเกียร์ (Transmission)", ["Manual", "Automatic"]
    )
    owner = st.selectbox(
        "เจ้าของมือที่ (Owner)",
        [
            "First Owner",
            "Second Owner",
            "Third Owner",
            "Fourth & Above Owner",
        ],
    )

    submitted = st.form_submit_button("🔍 ทำนายราคา")

if submitted:
    input_data = {
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
    }

    df_input = pd.DataFrame([input_data])

    for col, encoder in label_encoders.items():
        if col in df_input.columns:
            val = str(df_input[col].values[0])
            if val not in encoder.classes_:
                val = encoder.classes_[0]
            df_input[col] = encoder.transform([val])

    if hasattr(scaler, "feature_names_in_"):
        df_input = df_input[scaler.feature_names_in_]

    X_scaled = scaler.transform(df_input)
    X_input = np.expand_dims(X_scaled, axis=2)

    predicted_price = float(model.predict(X_input)[0][0])

    st.success("ทำนายผลสำเร็จ!")
    st.metric(
        label="💰 ราคาประเมินโดยประมาณ", value=f"{predicted_price:,.2f} บาท"
    )