# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle

def train_model():
    # Load preprocessed data
    data = pd.read_csv('../data/processed/preprocessed_student_data.csv')
    
    # Define target and features
    target_columns = ['Score', 'Grade', 'Teacher_Comments', 'Exam_Type']
    
    # Columns to drop
    to_drop = ['Score', 'Grade', 'Teacher_Comments', 'Exam_Type', 'Student_ID']
    
    # Check data types
    print(data.info())  # To check for any non-numeric columns
    
    # Make sure the features contain only numeric columns
    X = data.drop(to_drop, axis=1)
    y = data['Score']
    
    # Ensure that all feature columns are numeric
    non_numeric_cols = X.select_dtypes(include=['object']).columns.tolist()
    if len(non_numeric_cols) > 0:
        print(f"Non-numeric columns found: {non_numeric_cols}")
        
        # Apply LabelEncoder to encode non-numeric (categorical) columns
        label_encoder = LabelEncoder()
        
        # Iterate through each non-numeric column and apply label encoding
        for col in non_numeric_cols:
            X[col] = label_encoder.fit_transform(X[col])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred, squared=True)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Model - Mean Squared Error: {mse}')
    print(f'Model - R^2 Score: {r2}')
    
    # Save the trained model
    with open('../data/processed/student_performance_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved to '../data/processed/student_performance_model.pkl'")

if __name__ == '__main__':
    train_model()
