# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

def train_model():
    # Load preprocessed data
    data = pd.read_csv('../data/processed/preprocessed_student_data.csv')
    
    # Define target and features
    target_columns = ['Score', 'Grade', 'Teacher_Comments', 'Exam_Type']
    X = data.drop(target_columns, axis=1)
    y = data['Score']
    
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
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Model - Mean Squared Error: {mse}')
    print(f'Model - R^2 Score: {r2}')
    
    # Save the trained model
    with open('../data/processed/student_performance_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved to '../data/processed/student_performance_model.pkl'")

if __name__ == '__main__':
    train_model()