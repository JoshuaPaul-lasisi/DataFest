import pandas as pd

def load_data():
    attendance = pd.read_csv('../data/raw/Attendance_Records.csv')
    extra_cur = pd.read_csv('../data/raw/Extra_Curricular.csv')
    parents = pd.read_csv('../data/raw/parents_info.csv')
    performance = pd.read_csv('../data/raw/performance_records.csv')
    resource = pd.read_csv('../data/raw/resource_allocation.csv')
    students = pd.read_csv('../data/raw/student_demographics.csv')
    teachers = pd.read_csv('../data/raw/Teacher_info.csv')
    return attendance, extra_cur, parents, performance, resource, students, teachers

def merge_data():
    attendance, extra_cur, parents, performance, resource, students, teachers = load_data()
        
    # Merge all datasets
    data = students.merge(attendance, on='Student_ID', how='left')\
                   .merge(extra_cur, on='Student_ID', how='left')\
                   .merge(parents, on='Student_ID', how='left')\
                   .merge(performance, on='Student_ID', how='left')\
                   .merge(resource, on='Student_ID', how='left')\
                   .merge(teachers, on='Student_ID', how='left')
    
    # Save merged data
    data.to_csv('../data/processed/merged_student_data.csv', index=False)
    print("Merged data saved to '../data/processed/merged_student_data.csv'")

if __name__ == '__main__':
    merge_data()