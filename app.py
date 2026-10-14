import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Student Marks Prediction",
    page_icon="🎓",
    layout="centered"
)

# App Title
st.title("🎓 Student Marks Prediction App")
st.write(
    "Enter study hours and attendance "
    "to predict student marks using Machine Learning."
)

# Load Trained Model
@st.cache_resource
def load_model():
    return joblib.load("student_marks_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found! "
        "Please upload student_marks_model.pkl to GitHub."
    )
    st.stop()

# Student Details
st.header("Student Details")

study_hour = st.slider(
    "Daily Study Hours",
    min_value=1,
    max_value=12,
    value=5
)

attendance = st.slider(
    "Attendance (%)",
    min_value=10,
    max_value=100,
    value=75
)

# Prediction
if st.button("Predict Marks"):
    input_data = pd.DataFrame({
        "Study_hour": [study_hour],
        "attendance": [attendance]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"🎉 Predicted Marks: {prediction:.2f}"
    )

    st.write("Study Hours:", study_hour)
    st.write("Attendance:", f"{attendance}%")

st.caption("Built with Python, Pandas, Scikit-learn and Streamlit")
