import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="ML Assignment 2",
    page_icon="🤖"
)

st.title("🤖 Machine Learning Classification Dashboard")

uploaded_file = st.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset Uploaded Successfully")

    st.write("Dataset Preview")

    st.dataframe(df.head())