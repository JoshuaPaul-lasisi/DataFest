import streamlit as st
import pandas as pd
import pickle

# Load the preprocessed data
data = pd.read_csv('./data/processed/preprocessed_student_data.csv')

# Load the trained model
with open('./data/processed/student_performance_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app
st.title("Student Performance Prediction")

# Input: Allow users to select a student from the data
student_id = st.selectbox("Select a student", data['Student_ID'].unique())

# Show student data
student_data = data[data['Student_ID'] == student_id]
st.write("Student Data", student_data)

# Predict student performance
if st.button("Predict Performance"):
    prediction = model.predict(student_data.drop(['Score'], axis=1))  # Remove the target column 'Score'
    st.write(f"Predicted Performance: {prediction[0]}")