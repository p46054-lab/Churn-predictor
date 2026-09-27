import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("Customer Churn Predictor")
st.write("Fill in customer details to predict if they will churn (leave).")

user_input = {}
for col in columns:
    if col in encoders:
        options = list(encoders[col].classes_)
        choice = st.selectbox(col, options)
        user_input[col] = encoders[col].transform([choice])[0]
    else:
        user_input[col] = st.number_input(col, value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])[columns]
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Likely to churn (probability: {prob:.2%})")
    else:
        st.success(f"✅ Likely to stay (probability of churn: {prob:.2%})")
