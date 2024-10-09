import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def preprocess_data():
    # Load merged data
    data = pd.read_csv('../data/processed/merged_student_data.csv')
    
    # Identify numerical and categorical columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = data.select_dtypes(include=['object', 'bool']).columns
    
    # Fill numerical columns with mean
    data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())
    
    # Fill categorical columns with mode
    data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
    
    # Apply .infer_objects() to ensure appropriate dtype inference
    data[categorical_cols] = data[categorical_cols].infer_objects(copy=False)
    
    # One-hot encode categorical variables
    categorical_columns = ['Parents_Education_Level', 'Father_Occupation', 'Mother_Occupation', 
                           'Parental_Support', 'Activity_Name', 'Role', 'Impact_on_Academics', 
                           'Qualification', 'Status', 'Reason_for_Absence']

    categorical_columns = [col for col in categorical_columns if col in data.columns]

    if categorical_columns:  # Check if there are categorical columns to encode
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(data[categorical_columns])
        encoded_feature_names = encoder.get_feature_names_out(categorical_columns)
    
        # Create DataFrame with encoded features
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names)
    
        # Reset index for concatenation
        encoded_df.reset_index(drop=True, inplace=True)
        data.reset_index(drop=True, inplace=True)
    
        # Drop original categorical columns and concatenate encoded ones
        data = pd.concat([data.drop(categorical_columns, axis=1), encoded_df], axis=1)
    
    # Save preprocessed data
    data.to_csv('../data/processed/preprocessed_student_data.csv', index=False)
    print("Preprocessed data saved to '../data/processed/preprocessed_student_data.csv'")

if __name__ == '__main__':
    preprocess_data()
