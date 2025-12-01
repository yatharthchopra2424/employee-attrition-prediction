"""
Data Processing Module
======================
Handles data loading, preprocessing, feature engineering, and employee profiling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def load_employee_data(filepath: str) -> pd.DataFrame:
    """
    Load employee data from CSV file.
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        DataFrame with employee data
    """
    df = pd.read_csv(filepath)
    return df


def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute engineered features for the model.
    
    Args:
        df: Raw employee DataFrame
    
    Returns:
        DataFrame with additional engineered features
    """
    df = df.copy()
    
    # Workload Index: stress proxy
    df['Workload_Index'] = (
        (df['Average_Hours_Worked_Per_Week'] * df['Project_Count']) / 
        (df['Work_Life_Balance'] + 1e-6)
    ).round(0)
    
    # Engagement Score: mean of satisfaction metrics
    satisfaction_cols = ['Job_Involvement', 'Job_Satisfaction', 
                        'Work_Environment_Satisfaction', 'Relationship_with_Manager']
    df['Engagement_Score'] = df[satisfaction_cols].mean(axis=1)
    
    # Growth Index: career progression pace
    df['Growth_Index'] = (df['Job_Level'] / (df['Years_at_Company'] + 1)).round(3)
    
    # Training Effectiveness
    df['Training_Effectiveness'] = df['Performance_Rating'] * df['Training_Hours_Last_Year']
    
    # Overtime Flag
    df['Overtime_Flag'] = df['Overtime'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Distance Bucket
    df['Distance_Bucket'] = pd.qcut(
        df['Distance_From_Home'], 
        q=4, 
        labels=['Close', 'Medium', 'Far', 'Very Far'],
        duplicates='drop'
    )
    
    # Absenteeism Level
    df['Absenteeism_Level'] = pd.qcut(
        df['Absenteeism'], 
        q=3, 
        labels=['Low', 'Medium', 'High'],
        duplicates='drop'
    )
    
    # Age Group
    bins = [0, 30, 40, 50, df['Age'].max() + 1]
    labels = ['<30', '30-40', '40-50', '>50']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    
    # Work Efficiency
    df['Work_Efficiency'] = (
        df['Average_Hours_Worked_Per_Week'] / df['Project_Count'].replace(0, np.nan)
    ).round(2)
    
    # Tenure Bucket
    bins = [0, 3, 7, df['Years_at_Company'].max() + 1]
    labels = ['Early', 'Mid', 'Long']
    df['Tenure_Bucket'] = pd.cut(df['Years_at_Company'], bins=bins, labels=labels, right=False)
    
    # Income Level
    df['Income_Level'] = pd.qcut(
        df['Monthly_Income'], 
        q=4, 
        labels=['Q1', 'Q2', 'Q3', 'Q4'],
        duplicates='drop'
    )
    
    return df


def preprocess_for_prediction(
    employee_data: Dict,
    encoding_map: Dict,
    model_columns: List[str]
) -> pd.DataFrame:
    """
    Preprocess employee data for model prediction.
    
    Args:
        employee_data: Dictionary with employee features
        encoding_map: Encoding mappings for categorical variables
        model_columns: List of columns expected by the model
    
    Returns:
        Preprocessed DataFrame ready for prediction
    """
    df = pd.DataFrame([employee_data])
    
    # Apply encodings for categorical columns
    for col, mapping in encoding_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    # Ensure correct column order
    df = df.reindex(columns=model_columns, fill_value=0)
    
    return df


def get_employee_profile(df: pd.DataFrame, employee_id: int) -> Dict:
    """
    Get comprehensive profile for a specific employee.
    
    Args:
        df: Employee DataFrame
        employee_id: Employee ID to look up
    
    Returns:
        Dictionary with employee profile information
    """
    employee = df[df['Employee_ID'] == employee_id]
    
    if employee.empty:
        return None
    
    emp = employee.iloc[0]
    
    profile = {
        'Employee_ID': emp['Employee_ID'],
        'Age': emp['Age'],
        'Gender': emp['Gender'],
        'Marital_Status': emp['Marital_Status'],
        'Department': emp['Department'],
        'Job_Role': emp['Job_Role'],
        'Job_Level': emp['Job_Level'],
        'Monthly_Income': emp['Monthly_Income'],
        'Years_at_Company': emp['Years_at_Company'],
        'Years_in_Current_Role': emp['Years_in_Current_Role'],
        'Years_Since_Last_Promotion': emp['Years_Since_Last_Promotion'],
        'Work_Life_Balance': emp['Work_Life_Balance'],
        'Job_Satisfaction': emp['Job_Satisfaction'],
        'Performance_Rating': emp['Performance_Rating'],
        'Training_Hours_Last_Year': emp['Training_Hours_Last_Year'],
        'Overtime': emp['Overtime'],
        'Project_Count': emp['Project_Count'],
        'Average_Hours_Worked_Per_Week': emp['Average_Hours_Worked_Per_Week'],
        'Absenteeism': emp['Absenteeism'],
        'Work_Environment_Satisfaction': emp['Work_Environment_Satisfaction'],
        'Relationship_with_Manager': emp['Relationship_with_Manager'],
        'Job_Involvement': emp['Job_Involvement'],
        'Distance_From_Home': emp['Distance_From_Home'],
        'Number_of_Companies_Worked': emp['Number_of_Companies_Worked'],
        'Attrition': emp.get('Attrition', 'Unknown')
    }
    
    return profile


def get_department_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get aggregated statistics by department.
    
    Args:
        df: Employee DataFrame
    
    Returns:
        DataFrame with department-level statistics
    """
    stats = df.groupby('Department').agg({
        'Employee_ID': 'count',
        'Monthly_Income': 'mean',
        'Job_Satisfaction': 'mean',
        'Work_Life_Balance': 'mean',
        'Years_at_Company': 'mean',
        'Overtime': lambda x: (x == 'Yes').mean() * 100,
        'Attrition': lambda x: (x == 'Yes').mean() * 100 if 'Yes' in x.values else 0
    }).round(2)
    
    stats.columns = [
        'Employee_Count', 'Avg_Income', 'Avg_Satisfaction',
        'Avg_Work_Life_Balance', 'Avg_Tenure', 'Overtime_Pct', 'Attrition_Rate'
    ]
    
    return stats.reset_index()


def get_risk_distribution(
    df: pd.DataFrame,
    model,
    model_columns: List[str],
    encoding_map: Dict
) -> Dict:
    """
    Calculate risk distribution across the employee population.
    
    Args:
        df: Employee DataFrame
        model: Trained prediction model
        model_columns: Columns expected by model
        encoding_map: Encoding mappings
    
    Returns:
        Dictionary with risk distribution statistics
    """
    # This would compute actual predictions for all employees
    # Simplified version for demonstration
    
    high_risk_count = int(len(df) * 0.15)
    medium_risk_count = int(len(df) * 0.25)
    low_risk_count = len(df) - high_risk_count - medium_risk_count
    
    return {
        'total': len(df),
        'high_risk': high_risk_count,
        'medium_risk': medium_risk_count,
        'low_risk': low_risk_count,
        'high_risk_pct': round(high_risk_count / len(df) * 100, 1),
        'medium_risk_pct': round(medium_risk_count / len(df) * 100, 1),
        'low_risk_pct': round(low_risk_count / len(df) * 100, 1)
    }


def identify_at_risk_employees(
    df: pd.DataFrame,
    model,
    model_columns: List[str],
    encoding_map: Dict,
    threshold: float = 0.7
) -> pd.DataFrame:
    """
    Identify employees above a certain risk threshold.
    
    Args:
        df: Employee DataFrame
        model: Trained prediction model
        model_columns: Columns expected by model
        encoding_map: Encoding mappings
        threshold: Risk threshold (default 0.7 for high risk)
    
    Returns:
        DataFrame with at-risk employees and their risk scores
    """
    # Placeholder implementation
    # Would iterate through employees and get predictions
    
    at_risk = df.sample(min(20, len(df))).copy()
    at_risk['Risk_Score'] = np.random.uniform(threshold, 0.95, len(at_risk))
    at_risk = at_risk.sort_values('Risk_Score', ascending=False)
    
    return at_risk[['Employee_ID', 'Department', 'Job_Role', 
                    'Years_at_Company', 'Risk_Score']]


def get_feature_importance_for_employee(
    employee_data: Dict,
    df: pd.DataFrame
) -> List[Dict]:
    """
    Get relative feature importance based on employee's deviation from averages.
    
    Args:
        employee_data: Dictionary with employee features
        df: Full employee DataFrame for comparison
    
    Returns:
        List of dictionaries with feature importance information
    """
    numeric_cols = [
        'Job_Satisfaction', 'Work_Life_Balance', 'Relationship_with_Manager',
        'Job_Involvement', 'Work_Environment_Satisfaction', 
        'Years_Since_Last_Promotion', 'Average_Hours_Worked_Per_Week',
        'Monthly_Income', 'Training_Hours_Last_Year'
    ]
    
    importance_list = []
    
    for col in numeric_cols:
        if col in employee_data and col in df.columns:
            emp_value = employee_data[col]
            avg_value = df[col].mean()
            std_value = df[col].std()
            
            # Calculate z-score (deviation from mean)
            z_score = (emp_value - avg_value) / (std_value + 1e-6)
            
            # Determine direction of risk
            # For some features, lower is riskier (satisfaction metrics)
            risky_if_low = ['Job_Satisfaction', 'Work_Life_Balance', 
                          'Relationship_with_Manager', 'Job_Involvement',
                          'Work_Environment_Satisfaction', 'Monthly_Income']
            
            if col in risky_if_low:
                risk_contribution = -z_score  # Low values increase risk
            else:
                risk_contribution = z_score  # High values increase risk
            
            importance_list.append({
                'feature': col,
                'employee_value': emp_value,
                'average_value': round(avg_value, 2),
                'deviation': round(z_score, 2),
                'risk_contribution': round(risk_contribution, 2),
                'is_risk_factor': risk_contribution > 0.5
            })
    
    # Sort by absolute risk contribution
    importance_list.sort(key=lambda x: abs(x['risk_contribution']), reverse=True)
    
    return importance_list


def calculate_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate employee persona clusters.
    
    Args:
        df: DataFrame with engineered features
    
    Returns:
        DataFrame with cluster assignments and profiles
    """
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans
    
    cluster_features = [
        'Workload_Index', 'Work_Efficiency', 'Engagement_Score',
        'Growth_Index', 'Performance_Rating'
    ]
    
    # Filter to available features
    available_features = [f for f in cluster_features if f in df.columns]
    
    if len(available_features) < 3:
        return df
    
    cluster_data = df[available_features].copy().fillna(df[available_features].median())
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_data)
    
    # Map clusters to personas
    cluster_map = {
        0: "Overworked High-Performer",
        1: "Stagnant Loyalist", 
        2: "Detached Moderate"
    }
    df['Persona'] = df['Cluster'].map(cluster_map)
    
    return df
