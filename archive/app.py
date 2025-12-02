"""
Velorium Employee Retention Copilot
===================================
A comprehensive AI-powered retention assistant for managers to identify at-risk employees
and receive personalized, empathetic intervention recommendations.

"Data didn't make us colder. It made us listen." - Raghav Sethi, CEO Velorium Technologies
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os

# Local imports
from utils.encoding_map import ENCODING_MAP

# Try importing utilities (may fail if dependencies not installed yet)
try:
    from utils.shap_explainer import get_shap_explanation, get_top_risk_factors, generate_shap_summary
    from utils.genai_engine import (
        generate_retention_recommendation,
        generate_email_draft,
        generate_1on1_talking_points,
        generate_policy_suggestion,
        chat_with_copilot,
        configure_ai_engine
    )
    from utils.data_processor import (
        load_employee_data,
        preprocess_for_prediction,
        compute_engineered_features,
        get_employee_profile,
        get_department_stats,
        get_risk_distribution
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Velorium Retention Copilot",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

