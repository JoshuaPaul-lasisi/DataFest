import streamlit as st
import pandas as pd
import pickle
import os
import subprocess

# Define the paths to the processed files
merged_data_path = './data/processed/merged_student_data.csv'
preprocessed_data_path = './data/processed/preprocessed_student_data.csv'
model_path = './data/processed/student_performance_model.pkl'

# Run scripts sequentially
def run_script(script_name):
    try:
        subprocess.run(['python', script_name], check=True)
    except subprocess.CalledProcessError as e:
        st.error(f'Errorrunning {script_name}: {e}')

# Check merged data
if not os.path.exists(merged_data_path):
    st.info('Merging datasets...')
    run_script('./src/merge_data.py')

# Check preprocessed data
if not os.path.exists(preprocessed_data_path):
    st.info('Processing the data...')
    run_script('./src/preprocessing.py')

# Check model file
if not os.path.exists(model_path):
    st.info('Building the model...')
    run_script('./src/model.py')

# Load the preprocessed data
data = pd.read_csv(preprocessed_data_path)

# Load the trained model
with open(model_path, 'rb') as f:
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