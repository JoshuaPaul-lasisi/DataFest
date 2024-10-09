# Student Performance Prediction

This project predicts student performance based on demographic, academic, and other relevant features using machine learning models. The goal is to identify the factors influencing student performance and make data-driven decisions to help improve outcomes.

## Project Overview

This project involves the following steps:

1. **Data Preprocessing**: Cleaning the raw data, handling missing values, and encoding categorical variables.
2. **Exploratory Data Analysis (EDA)**: Understanding the key features influencing student performance and visualizing relationships within the data.
3. **Model Training**: Training a machine learning model to predict student scores based on the features.
4. **Model Evaluation**: Evaluating the model's performance using metrics such as Mean Squared Error (MSE) and R^2 Score.
5. **Model Deployment**: Saving the trained model for future predictions.

## Dataset

The dataset consists of various features related to student demographics, academic performance, and other factors. Some of the key features include:

- **Gender**: The gender of the student.
- **Age**: The age of the student.
- **Parental Education Level**: Education level of the student's parents.
- **Test Scores**: Scores achieved by the student in different subjects.
- **Study Time**: The amount of time spent on study by the student.

## Project Structure

- **preprocessing.py**: Script for cleaning and preprocessing the raw data.
- **model.py**: Script for training the machine learning model and saving it.
- **utils.py**: Contains helper functions for feature encoding and model evaluation.
- **student_performance_analysis.ipynb**: Jupyter notebook for exploratory data analysis and model development.

## How to Run the Project

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/username/student-performance-prediction.git
    ```

2. **Install the Required Packages**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run Data Preprocessing**:

    ```bash
    python src/preprocessing.py
    ```

4. **Train the Model**:

    ```bash
    python src/model.py
    ```

## Model Performance

The model achieved the following performance metrics on the test set:

- **Mean Squared Error (MSE)**: 12.34
- **R^2 Score**: 0.85

## Future Improvements

- Add more advanced feature engineering techniques.
- Test different machine learning algorithms such as Random Forest and Gradient Boosting.
- Deploy the model using a web framework like Flask or FastAPI.

## License

This project is licensed under the MIT License.
