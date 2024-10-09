import streamlit as st
import pandas as pd
import pickle
import os

from merge_data import merge_data
from preprocessing import preprocess_data
from model import train_model

def load_model():
    with open('../data/processed/student_performance_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def load_data():
    return pd.read_csv('../data/processed/preprocessed_student_data.csv')

def run_app():
    st.title("Student Performance Prediction")
    
    # Load data and model
    data = load_data()
    model = load_model()
    
    st.subheader("Preprocessed Dataset")
    st.write(data.head())
    
    # Select a student or input features manually
    st.subheader("Predict Student Score")
    
    # For simplicity, allow user to input some features manually
    # Alternatively, you could allow selecting a student and use their features
    
    # Example inputs (you need to adjust based on actual feature names)
    st.write("Please enter the following details to predict the score:")
    
    # Example feature selection (you should replace with actual features)
    # Here, we'll list some columns from the preprocessed data
    feature_cols = data.columns.tolist()
    feature_cols.remove('Score')  # Remove target
    
    user_input = {}
    for feature in feature_cols:
        if data[feature].dtype in ['float64', 'int64']:
            user_input[feature] = st.number_input(feature, value=float(data[feature].mean()))
        else:
            unique_vals = data[feature].unique().tolist()
            user_input[feature] = st.selectbox(feature, unique_vals)
    
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Ensure the input features match the model's expected features
    missing_cols = set(data.drop(['Score'], axis=1).columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  # or appropriate default value
    
    input_df = input_df[data.drop(['Score'], axis=1).columns]  # Reorder columns
    
    # Predict
    if st.button("Predict Score"):
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Score: {prediction:.2f}")

if __name__ == '__main__':
    run_app()