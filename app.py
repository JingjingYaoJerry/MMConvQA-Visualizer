import streamlit as st

from data_loader import load_data

st.title("MMCoQA Explorer")
st.write("Hi team!")

st.header("Data Sample")
data = load_data()
st.json(data[1])  # Display the second data entry