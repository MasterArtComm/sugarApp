import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests

st.title("SugarScan Application ")

# โหลดโมเดล
with open("xgb_sugar_model.pkl", "rb") as file:
    model = pickle.load(file)

API_URL = "https://techno.varee.ac.th/users/sugarApp/get_data.php"  # แก้เป็น URL ของคุณ

@st.cache_data(ttl=600)
def load_data():
    try:
        response = requests.get(API_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"ไม่สามารถดึงข้อมูลจาก API ได้: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("ไม่มีข้อมูลให้แสดง")
else:
    # แปลงชนิดข้อมูลให้ถูกต้อง
    df["rs_ro"] = pd.to_numeric(df["rs_ro"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    # เตรียมข้อมูลสำหรับทำนาย (feature1=rs_ro, feature2=bmi)
    X = df[["rs_ro", "bmi"]].values

    # ทำนายทีละแถว
    predictions = model.predict(X)

    # ใส่คอลัมน์ผลลัพธ์ลง DataFrame
    df["prediction"] = predictions

    st.subheader("ประวัติการตรวจจากลมหายใจ")
    st.dataframe(df)
