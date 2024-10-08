import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load preprocessed data
data = pd.read_csv('../data/processed/preprocessed_student_data.csv')

# Select features and target variable (you need to define your target variable)
X = data.drop(['Score_y'], axis=1)  # Example: using all columns except 'Score'
y = data['Score_y']  # Assuming 'Score' is the target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model to a file
with open('../data/processed/student_performance_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to '../data/processed/student_performance_model.pkl'")
