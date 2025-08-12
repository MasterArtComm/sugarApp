import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests

st.title("SugarScan Application ")

# โหลดโมเดล
with open("xgb_sugar_model.pkl", "rb") as file:
    model = pickle.load(file)

API_DATA_URL = "https://techno.varee.ac.th/users/sugarApp/get_data.php"  # URL ดึงข้อมูล
API_UPDATE_URL = "https://techno.varee.ac.th/users/sugarApp/update_prediction.php"  # URL อัปเดตผลทำนาย (ต้องสร้าง)

@st.cache_data(ttl=600)
def load_data():
    try:
        response = requests.get(API_DATA_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"ไม่สามารถดึงข้อมูลจาก API ได้: {e}")
        return pd.DataFrame()

def update_prediction(measurement_id, predicted_value):
    try:
        payload = {
            "measurement_id": measurement_id,
            "predicted_glucose_mgdl": float(predicted_value)
        }
        res = requests.post(API_UPDATE_URL, data=payload, timeout=5)
        if res.status_code == 200:
            return True
        else:
            st.warning(f"อัปเดต measurement_id={measurement_id} ไม่สำเร็จ: {res.text}")
            return False
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอัปเดต measurement_id={measurement_id}: {e}")
        return False

df = load_data()

if df.empty:
    st.warning("ไม่มีข้อมูลให้แสดง")
else:
    df["rs_ro"] = pd.to_numeric(df["rs_ro"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    X = df[["rs_ro", "bmi"]].values
    predictions = model.predict(X)
    df["prediction"] = predictions

    st.subheader("ประวัติการตรวจจากลมหายใจ")
    st.dataframe(df)

    st.subheader("กำลังอัปเดตผลทำนายลงฐานข้อมูล...")
    for idx, row in df.iterrows():
        measurement_id = row["measurement_id"]
        predicted_value = row["prediction"]
        success = update_prediction(measurement_id, predicted_value)
        if success:
            st.write(f"อัปเดต measurement_id={measurement_id} สำเร็จ")
        else:
            st.write(f"อัปเดต measurement_id={measurement_id} ล้มเหลว")
st.write('<a href="https://techno.varee.ac.th/users/sugarApp/profile.php" target="_blank">Profile</a>', unsafe_allow_html=True)
st.write('<a href="https://techno.varee.ac.th/users/sugarApp/measurements_page.php" target="_blank">กราฟ</a>', unsafe_allow_html=True)