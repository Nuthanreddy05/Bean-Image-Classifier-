import streamlit as st
import pandas as pd
from src.predict import predict

st.title("🌱 Dry Bean Type Classifier")

uploaded = st.file_uploader("Upload Dry Bean CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Sample Input:", df.head())
    labels = predict(df)
    df['Predicted Class'] = labels
    st.write(df)

    st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")
