import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests

@st.cache_data(ttl=300)
def load_data():
    response = requests.get("https://techno.varee.ac.th/users/sugarApp/get_data.php")
    data = response.json()
    df = pd.DataFrame(data)
    return df.tail(500)  # ลดข้อมูลเหลือ 500 แถวล่าสุด

@st.cache_data(ttl=300)
def predict(df, model):
    X = df[["rs_ro", "bmi"]].astype(float).values
    preds = model.predict(X)
    df = df.copy()
    df["prediction"] = preds
    return df

with open("xgb_sugar_model.pkl", "rb") as f:
    model = pickle.load(f)

df = load_data()

if df.empty:
    st.warning("ไม่มีข้อมูล")
else:
    df = predict(df, model)
    st.dataframe(df)
    st.line_chart(df["prediction"])
