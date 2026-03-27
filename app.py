import streamlit as st
import pandas as pd
import joblib

st.title("Attrition Predictor")

model = joblib.load("attrition_model.pkl")

age = st.slider("Age", 18, 60, 30)
income = st.number_input("Income", 1000, 20000, 5000)
satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
balance = st.slider("Work Life Balance", 1, 4, 3)
years = st.slider("Years at Company", 0, 40, 5)
distance = st.slider("Distance", 1, 50, 10)
overtime = st.selectbox("Overtime", ["Yes", "No"])

overtime = 1 if overtime == "Yes" else 0

data = pd.DataFrame({
    'Age': [age],
    'MonthlyIncome': [income],
    'JobSatisfaction': [satisfaction],
    'WorkLifeBalance': [balance],
    'YearsAtCompany': [years],
    'DistanceFromHome': [distance],
    'OverTime_Yes': [overtime]
})

if st.button("Predict"):
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    st.write("Prediction:", "Leave" if pred == 1 else "Stay")
    st.write("Probability:", prob)
