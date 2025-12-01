"""
SHAP Explainer Module
=====================
Provides SHAP-based explanations for attrition risk predictions.
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Tuple, Optional


def get_shap_explanation(
    model,
    employee_data: pd.DataFrame,
    feature_names: List[str]
) -> Dict:
    """
    Generate SHAP explanation for an employee's attrition risk prediction.
    
    Args:
        model: Trained model (supports tree-based models)
        employee_data: DataFrame with employee features
        feature_names: List of feature names
    
    Returns:
        Dictionary containing SHAP values and interpretations
    """
    try:
        # Create SHAP explainer
        if hasattr(model, 'named_estimators_'):
            # Voting classifier - use XGBoost component
            base_model = model.named_estimators_.get('xgb', list(model.named_estimators_.values())[0])
        else:
            base_model = model
        
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(employee_data)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get values for positive class
        
        return {
            'shap_values': shap_values,
            'expected_value': explainer.expected_value,
            'feature_names': feature_names,
            'success': True
        }
    except Exception as e:
        return {
            'shap_values': None,
            'expected_value': None,
            'error': str(e),
            'success': False
        }


def get_top_risk_factors(
    shap_values: np.ndarray,
    feature_names: List[str],
    feature_values: pd.Series,
    top_n: int = 6
) -> List[Dict]:
    """
    Get top risk factors based on SHAP values.
    
    Args:
        shap_values: SHAP values for the employee
        feature_names: List of feature names
        feature_values: Actual feature values for the employee
        top_n: Number of top factors to return
    
    Returns:
        List of dictionaries with risk factor information
    """
    # Create dataframe of SHAP values
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values.flatten() if shap_values.ndim > 1 else shap_values,
        'feature_value': feature_values.values
    })
    
    # Sort by absolute SHAP value
    shap_df['abs_shap'] = shap_df['shap_value'].abs()
    shap_df = shap_df.sort_values('abs_shap', ascending=False).head(top_n)
    
    # Create human-readable risk factors
    risk_factors = []
    for _, row in shap_df.iterrows():
        direction = "increasing" if row['shap_value'] > 0 else "decreasing"
        impact = "negative" if row['shap_value'] > 0 else "positive"  # For attrition, positive SHAP = more likely to leave
        
        risk_factors.append({
            'factor': row['feature'],
            'value': row['feature_value'],
            'shap_value': row['shap_value'],
            'impact': impact,
            'direction': direction,
            'magnitude': abs(row['shap_value'])
        })
    
    return risk_factors


def get_factor_interpretation(factor_name: str, factor_value: float, shap_value: float) -> str:
    """
    Generate human-readable interpretation of a risk factor.
    
    Args:
        factor_name: Name of the feature
        factor_value: Value of the feature
        shap_value: SHAP value for the feature
    
    Returns:
        Human-readable interpretation string
    """
    interpretations = {
        'Work_Life_Balance': {
            'low': "Poor work-life balance is significantly contributing to attrition risk",
            'high': "Good work-life balance is helping reduce attrition risk"
        },
        'Years_Since_Last_Promotion': {
            'low': "Recent promotion is positively impacting retention",
            'high': "Extended time without promotion is increasing disengagement risk"
        },
        'Overtime_Flag': {
            'low': "No overtime work is supporting healthy engagement",
            'high': "Regular overtime is contributing to burnout risk"
        },
        'Relationship_with_Manager': {
            'low': "Weak manager relationship is a significant risk factor",
            'high': "Strong manager relationship is a protective factor"
        },
        'Job_Satisfaction': {
            'low': "Low job satisfaction is driving attrition risk",
            'high': "High job satisfaction is supporting retention"
        },
        'Engagement_Score': {
            'low': "Low engagement is a warning sign for potential departure",
            'high': "High engagement indicates strong organizational commitment"
        },
        'Workload_Index': {
            'low': "Manageable workload supports employee wellbeing",
            'high': "High workload index suggests potential burnout"
        },
        'Growth_Index': {
            'low': "Slow career growth may be causing frustration",
            'high': "Good career progression is supporting retention"
        }
    }
    
    # Determine if value is low or high based on typical thresholds
    is_high = shap_value > 0  # Positive SHAP means increases attrition risk
    
    if factor_name in interpretations:
        return interpretations[factor_name]['high' if is_high else 'low']
    else:
        impact = "increasing" if is_high else "decreasing"
        return f"{factor_name} (value: {factor_value:.2f}) is {impact} attrition risk"


def generate_shap_summary(
    risk_factors: List[Dict],
    employee_id: str
) -> str:
    """
    Generate a narrative summary of SHAP analysis.
    
    Args:
        risk_factors: List of risk factor dictionaries
        employee_id: Employee identifier
    
    Returns:
        Narrative summary string
    """
    negative_factors = [f for f in risk_factors if f['impact'] == 'negative']
    positive_factors = [f for f in risk_factors if f['impact'] == 'positive']
    
    summary = f"## Risk Factor Analysis for Employee {employee_id}\n\n"
    
    if negative_factors:
        summary += "### 🔴 Factors Increasing Attrition Risk:\n"
        for i, factor in enumerate(negative_factors, 1):
            interpretation = get_factor_interpretation(
                factor['factor'], 
                factor['value'], 
                factor['shap_value']
            )
            summary += f"{i}. **{factor['factor']}** (Value: {factor['value']:.2f})\n"
            summary += f"   - {interpretation}\n"
            summary += f"   - Impact magnitude: {factor['magnitude']:.3f}\n\n"
    
    if positive_factors:
        summary += "### 🟢 Protective Factors:\n"
        for i, factor in enumerate(positive_factors, 1):
            interpretation = get_factor_interpretation(
                factor['factor'], 
                factor['value'], 
                factor['shap_value']
            )
            summary += f"{i}. **{factor['factor']}** (Value: {factor['value']:.2f})\n"
            summary += f"   - {interpretation}\n\n"
    
    return summary
