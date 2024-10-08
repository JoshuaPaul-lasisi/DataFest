import pandas as pd

# Load datasets
attendance = pd.read_csv('./data/raw/Attendance_Records.csv')
extra_cur = pd.read_csv('./data/raw/Extra_Curricular.csv')
new_performance = pd.read_csv('./data/raw/new_performance_records.csv')
parents = pd.read_csv('./data/raw/parents_info.csv')
performance = pd.read_csv('./data/raw/performance_records.csv')
resource = pd.read_csv('./data/raw/resource_allocation.csv')
students = pd.read_csv('./data/raw/student_demographics.csv')
teachers = pd.read_csv('./data/raw/Teacher_info.csv')

# Merge datasets
data = students.merge(attendance, on='Student_ID', how='left')\
               .merge(extra_cur, on='Student_ID', how='left')\
               .merge(new_performance, on='Student_ID', how='left')\
               .merge(parents, on='Student_ID', how='left')\
               .merge(performance, on='Student_ID', how='left')\
               .merge(resource, on='Student_ID', how='left')\
               .merge(teachers, on='Student_ID', how='left')

# Save merged data
data.to_csv('./data/processed/merged_student_data.csv', index=False)

print("Merged data saved to './data/processed/merged_student_data.csv'")
