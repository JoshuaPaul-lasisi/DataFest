import pandas as pd

# Load the merged data
data = pd.read_csv('./data/processed/merged_student_data.csv')

# Handle missing values (you can expand this as needed)
data.fillna('Unknown', inplace=True)

# Save preprocessed data
data.to_csv('./data/processed/preprocessed_student_data.csv', index=False)

print("Preprocessed data saved to './data/processed/preprocessed_student_data.csv'")
