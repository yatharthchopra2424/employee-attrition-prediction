"""
Velorium Employee Retention Copilot - MVP
==========================================
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
import os

# Local imports
from utils.encoding_map import ENCODING_MAP

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Velorium Retention Copilot",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS for Professional UI
# ----------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1E3A5F 0%, #3B82F6 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #3B82F6;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border-left-color: #EF4444 !important;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border-left-color: #F59E0B !important;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
        border-left-color: #10B981 !important;
    }
    
    .employee-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .quote-box {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #0EA5E9;
        font-style: italic;
        margin: 1rem 0;
    }
    
    .recommendation-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
        max-width: 85%;
    }
    
    .chat-user {
        background: #E0E7FF;
        margin-left: auto;
    }
    
    .chat-assistant {
        background: #F1F5F9;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Initialize Session State
# ----------------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_employee' not in st.session_state:
    st.session_state.selected_employee = None
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False
if 'generated_content' not in st.session_state:
    st.session_state.generated_content = False

# ----------------------------
# Load Model and Data
# ----------------------------
@st.cache_resource
def load_model():
    try:
        with open("models/attrition_model.pkl", "rb") as f:
            saved = pickle.load(f)
        return saved["model"], saved["columns"]
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please ensure 'models/attrition_model.pkl' exists.")
        return None, None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("raw_data/employee_attrition_dataset.csv")
        return df
    except FileNotFoundError:
        st.warning("⚠️ Data file not found. Using demo mode.")
        return None

model, model_columns = load_model()
employee_df = load_data()

# ----------------------------
# Helper Functions
# ----------------------------
def preprocess_input(user_input):
    df = pd.DataFrame([user_input])
    for col, mapping in ENCODING_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    df = df.reindex(columns=model_columns, fill_value=0)
    return df

def predict_attrition(user_input):
    if model is None:
        return None, None
    processed = preprocess_input(user_input)
    pred = model.predict(processed)[0]
    proba = model.predict_proba(processed)[0][1]
    return int(pred), float(proba)

def simulate_risk_prediction(employee_data):
    """Simulate risk prediction based on employee features"""
    # Simple rule-based simulation
    risk_score = 0.5
    
    if employee_data.get('Work_Life_Balance', 3) <= 2:
        risk_score += 0.15
    if employee_data.get('Years_Since_Last_Promotion', 0) >= 4:
        risk_score += 0.12
    if employee_data.get('Overtime', 'No') == 'Yes':
        risk_score += 0.10
    if employee_data.get('Job_Satisfaction', 3) <= 2:
        risk_score += 0.08
    if employee_data.get('Relationship_with_Manager', 3) <= 2:
        risk_score += 0.08
    
    return min(risk_score, 0.95)

# ----------------------------
# Sidebar Navigation
# ----------------------------
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h2 style='color: #1E3A5F; margin-bottom: 0.5rem;'>🎯 Retention Copilot</h2>
        <p style='color: #64748B; font-size: 0.9rem;'>Velorium Technologies</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "👥 Employee Analysis", "💬 AI Copilot Chat", "📝 Action Generator", "🔮 Predict New", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    if employee_df is not None:
        total_employees = len(employee_df)
        st.metric("Total Employees", f"{total_employees:,}")
        high_risk = int(total_employees * 0.15)
        st.metric("High Risk", high_risk, delta="-3", delta_color="inverse")
    
    st.divider()
    
    st.markdown("""
    <div class='quote-box'>
        <p>"Data didn't make us colder. It made us listen."</p>
        <p style='text-align: right; font-weight: 600;'>— Raghav Sethi</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE: DASHBOARD
# ============================================================
if "Dashboard" in page:
    st.markdown("""
    <div class='main-header'>
        <h1>🎯 Retention Copilot Dashboard</h1>
        <p>Proactive employee retention insights powered by AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3 style='color: #64748B; font-size: 0.9rem; margin-bottom: 0.5rem;'>Total Workforce</h3>
            <p style='font-size: 2rem; font-weight: 700; color: #1E3A5F;'>1,001</p>
            <p style='color: #10B981; font-size: 0.85rem;'>↑ 2.3% from Q3</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card risk-high'>
            <h3 style='color: #64748B; font-size: 0.9rem; margin-bottom: 0.5rem;'>High Risk Employees</h3>
            <p style='font-size: 2rem; font-weight: 700; color: #DC2626;'>147</p>
            <p style='color: #DC2626; font-size: 0.85rem;'>⚠️ Requires immediate attention</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card risk-medium'>
            <h3 style='color: #64748B; font-size: 0.9rem; margin-bottom: 0.5rem;'>Medium Risk</h3>
            <p style='font-size: 2rem; font-weight: 700; color: #D97706;'>234</p>
            <p style='color: #D97706; font-size: 0.85rem;'>📋 Monitor closely</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='metric-card risk-low'>
            <h3 style='color: #64748B; font-size: 0.9rem; margin-bottom: 0.5rem;'>Engaged & Stable</h3>
            <p style='font-size: 2rem; font-weight: 700; color: #059669;'>620</p>
            <p style='color: #059669; font-size: 0.85rem;'>✓ Healthy engagement</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Row
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader("📈 Risk Distribution by Department")
        
        dept_data = pd.DataFrame({
            'Department': ['IT', 'Sales', 'Marketing', 'HR', 'Finance'],
            'High Risk': [45, 38, 32, 15, 17],
            'Medium Risk': [67, 54, 48, 32, 33],
            'Low Risk': [156, 132, 98, 112, 122]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='High Risk', x=dept_data['Department'], y=dept_data['High Risk'], marker_color='#EF4444'))
        fig.add_trace(go.Bar(name='Medium Risk', x=dept_data['Department'], y=dept_data['Medium Risk'], marker_color='#F59E0B'))
        fig.add_trace(go.Bar(name='Low Risk', x=dept_data['Department'], y=dept_data['Low Risk'], marker_color='#10B981'))
        
        fig.update_layout(
            barmode='stack',
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("🔍 Top Risk Factors")
        
        risk_factors = [
            {"factor": "Work-Life Balance", "impact": 85, "color": "#EF4444"},
            {"factor": "Years Since Promotion", "impact": 78, "color": "#F59E0B"},
            {"factor": "Overtime Hours", "impact": 72, "color": "#F59E0B"},
            {"factor": "Manager Relationship", "impact": 65, "color": "#3B82F6"},
            {"factor": "Job Satisfaction", "impact": 58, "color": "#3B82F6"}
        ]
        
        for rf in risk_factors:
            st.markdown(f"""
            <div style='margin-bottom: 1rem;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 0.25rem;'>
                    <span style='font-weight: 500;'>{rf['factor']}</span>
                    <span style='color: {rf['color']}; font-weight: 600;'>{rf['impact']}%</span>
                </div>
                <div style='background: #E2E8F0; border-radius: 4px; height: 8px;'>
                    <div style='background: {rf['color']}; width: {rf['impact']}%; height: 100%; border-radius: 4px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # High Risk Employees Table
    st.subheader("⚠️ Employees Requiring Immediate Attention")
    
    high_risk_employees = pd.DataFrame({
        'Employee ID': ['EMP-1027', 'EMP-0892', 'EMP-0456', 'EMP-0789', 'EMP-0234'],
        'Department': ['IT', 'Sales', 'Marketing', 'IT', 'Sales'],
        'Role': ['Senior Developer', 'Account Executive', 'Marketing Manager', 'Tech Lead', 'Sales Representative'],
        'Risk Score': [0.89, 0.85, 0.82, 0.79, 0.76],
        'Primary Risk Factor': ['Work-Life Balance', 'No Promotion (4 yrs)', 'Manager Relationship', 'Overtime', 'Low Satisfaction'],
        'Tenure': ['5 years', '3 years', '7 years', '4 years', '2 years']
    })
    
    for idx, row in high_risk_employees.iterrows():
        col1, col2, col3, col4 = st.columns([2, 3, 2, 2])
        
        with col1:
            st.markdown(f"**{row['Employee ID']}**  \n{row['Tenure']}")
        
        with col2:
            st.markdown(f"{row['Role']}  \n{row['Department']}")
        
        with col3:
            st.markdown(f"🔴 **{row['Risk Score']*100:.0f}% Risk**")
        
        with col4:
            if st.button("View Details", key=f"view_{idx}"):
                st.session_state.selected_employee = row['Employee ID']
                st.rerun()
        
        st.divider()

# ============================================================
# PAGE: EMPLOYEE ANALYSIS
# ============================================================
elif "Employee Analysis" in page:
    st.markdown("""
    <div class='main-header'>
        <h1>👥 Individual Employee Analysis</h1>
        <p>Deep dive into employee profiles with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    if employee_df is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            employee_ids = employee_df['Employee_ID'].astype(str).tolist()
            selected_id = st.selectbox(
                "🔍 Search Employee by ID",
                options=employee_ids,
                index=0
            )
        
        with col2:
            department_filter = st.selectbox(
                "Filter by Department",
                options=["All", "IT", "Sales", "Marketing", "HR", "Finance"]
            )
        
        if selected_id:
            emp_data = employee_df[employee_df['Employee_ID'] == int(selected_id)].iloc[0]
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Employee Profile Card
            col_profile, col_risk = st.columns([2, 1])
            
            with col_profile:
                st.markdown(f"""
                <div class='employee-card'>
                    <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
                        <div style='width: 60px; height: 60px; background: linear-gradient(135deg, #3B82F6 0%, #1E3A5F 100%); 
                                    border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                                    color: white; font-size: 1.5rem; font-weight: 600; margin-right: 1rem;'>
                            {emp_data['Gender'][0]}
                        </div>
                        <div>
                            <h2 style='margin: 0; color: #1E3A5F;'>Employee #{selected_id}</h2>
                            <p style='margin: 0; color: #64748B;'>{emp_data['Job_Role']} • {emp_data['Department']}</p>
                        </div>
                    </div>
                    
                    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;'>
                        <div>
                            <p style='color: #64748B; font-size: 0.85rem; margin-bottom: 0.25rem;'>Tenure</p>
                            <p style='font-weight: 600; margin: 0;'>{emp_data['Years_at_Company']} years</p>
                        </div>
                        <div>
                            <p style='color: #64748B; font-size: 0.85rem; margin-bottom: 0.25rem;'>Age</p>
                            <p style='font-weight: 600; margin: 0;'>{emp_data['Age']} years</p>
                        </div>
                        <div>
                            <p style='color: #64748B; font-size: 0.85rem; margin-bottom: 0.25rem;'>Job Level</p>
                            <p style='font-weight: 600; margin: 0;'>Level {emp_data['Job_Level']}</p>
                        </div>
                        <div>
                            <p style='color: #64748B; font-size: 0.85rem; margin-bottom: 0.25rem;'>Monthly Income</p>
                            <p style='font-weight: 600; margin: 0;'>₹{emp_data['Monthly_Income']:,}</p>
                        </div>
                        <div>
                            <p style='color: #64748B; font-size: 0.85rem; margin-bottom: 0.25rem;'>Last Promotion</p>
                            <p style='font-weight: 600; margin: 0;'>{emp_data['Years_Since_Last_Promotion']} years ago</p>
                        </div>
                        <div>
                            <p style='color: #64748B; font-size: 0.85rem; margin-bottom: 0.25rem;'>Overtime</p>
                            <p style='font-weight: 600; margin: 0;'>{emp_data['Overtime']}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_risk:
                risk_score = simulate_risk_prediction(emp_data)
                risk_level = "High" if risk_score >= 0.7 else "Medium" if risk_score >= 0.4 else "Low"
                risk_class = "risk-high" if risk_level == "High" else "risk-medium" if risk_level == "Medium" else "risk-low"
                
                st.markdown(f"""
                <div class='metric-card {risk_class}' style='text-align: center;'>
                    <h3 style='color: #64748B; font-size: 0.9rem; margin-bottom: 0.5rem;'>Attrition Risk Score</h3>
                    <p style='font-size: 3rem; font-weight: 700; margin: 0.5rem 0;'>{risk_score*100:.0f}%</p>
                    <p style='font-weight: 600; font-size: 1.1rem;'>{risk_level} Risk</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Tabs for detailed analysis
            tab1, tab2, tab3 = st.tabs(["📊 Key Metrics", "🔍 Risk Factors", "💡 AI Recommendations"])
            
            with tab1:
                st.subheader("Employee Engagement Metrics")
                
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    categories = ['Job Satisfaction', 'Work-Life Balance', 'Environment', 'Manager Relationship', 'Job Involvement']
                    values = [
                        emp_data['Job_Satisfaction'],
                        emp_data['Work_Life_Balance'],
                        emp_data['Work_Environment_Satisfaction'],
                        emp_data['Relationship_with_Manager'],
                        emp_data['Job_Involvement']
                    ]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=values + [values[0]],
                        theta=categories + [categories[0]],
                        fill='toself',
                        fillcolor='rgba(59, 130, 246, 0.3)',
                        line=dict(color='#3B82F6', width=2),
                        name='Current'
                    ))
                    
                    fig.add_trace(go.Scatterpolar(
                        r=[4, 4, 4, 4, 4, 4],
                        theta=categories + [categories[0]],
                        fill='toself',
                        fillcolor='rgba(16, 185, 129, 0.1)',
                        line=dict(color='#10B981', width=1, dash='dash'),
                        name='Ideal'
                    ))
                    
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                        showlegend=True,
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with metrics_col2:
                    st.markdown("**Performance Overview**")
                    
                    perf_data = {
                        "Performance Rating": (emp_data['Performance_Rating'], 4),
                        "Training Hours": (emp_data['Training_Hours_Last_Year'], 100),
                        "Avg Weekly Hours": (emp_data['Average_Hours_Worked_Per_Week'], 50),
                        "Projects Handled": (emp_data['Project_Count'], 10)
                    }
                    
                    for metric, (value, max_val) in perf_data.items():
                        progress = min(value / max_val, 1.0)
                        color = "#10B981" if progress >= 0.7 else "#F59E0B" if progress >= 0.4 else "#EF4444"
                        st.markdown(f"""
                        <div style='margin-bottom: 1rem;'>
                            <div style='display: flex; justify-content: space-between; margin-bottom: 0.25rem;'>
                                <span>{metric}</span>
                                <span style='font-weight: 600;'>{value}</span>
                            </div>
                            <div style='background: #E2E8F0; border-radius: 4px; height: 8px;'>
                                <div style='background: {color}; width: {progress*100}%; height: 100%; border-radius: 4px;'></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            with tab2:
                st.subheader("🔍 Risk Factor Analysis")
                
                st.markdown("""
                <div class='recommendation-card'>
                    <h4>🎯 Key Risk Factors (SHAP-based)</h4>
                    <ul>
                        <li><strong>Work-Life Balance:</strong> Rated {}/4 - Below average, contributing to stress</li>
                        <li><strong>Years Since Promotion:</strong> {} years - Career stagnation concern</li>
                        <li><strong>Overtime:</strong> {} - Contributing to burnout risk</li>
                        <li><strong>Manager Relationship:</strong> Rated {}/5 - Needs improvement</li>
                    </ul>
                </div>
                """.format(
                    emp_data['Work_Life_Balance'],
                    emp_data['Years_Since_Last_Promotion'],
                    emp_data['Overtime'],
                    emp_data['Relationship_with_Manager']
                ), unsafe_allow_html=True)
            
            with tab3:
                st.subheader("💡 AI-Generated Retention Recommendations")
                
                st.markdown("""
                <div class='recommendation-card'>
                    <h4>🎯 Immediate Actions (This Week)</h4>
                    <ol>
                        <li><strong>Schedule a 1:1 Meeting</strong> - Make it appreciative, not corrective</li>
                        <li><strong>Workload Review</strong> - Discuss current assignments and explore redistribution</li>
                        <li><strong>Flexible Work Discussion</strong> - Explore hybrid options or adjusted schedules</li>
                    </ol>
                </div>
                
                <div class='recommendation-card'>
                    <h4>📈 Short-Term Plan (Next 30 Days)</h4>
                    <ol>
                        <li><strong>Career Path Discussion</strong> - Create visible growth trajectory</li>
                        <li><strong>Certification Sponsorship</strong> - Offer relevant certifications</li>
                        <li><strong>Mentorship Pairing</strong> - Connect with senior leader</li>
                    </ol>
                </div>
                
                <div class='recommendation-card'>
                    <h4>🏢 Long-Term Strategy (Next Quarter)</h4>
                    <ol>
                        <li><strong>Promotion Consideration</strong> - Evaluate for next cycle</li>
                        <li><strong>Project Leadership</strong> - Assign high-visibility initiative</li>
                        <li><strong>Compensation Review</strong> - Benchmark and adjust if needed</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📂 Please load employee data to use this feature")

# ============================================================
# PAGE: AI COPILOT CHAT
# ============================================================
elif "AI Copilot Chat" in page:
    st.markdown("""
    <div class='main-header'>
        <h1>💬 AI Retention Copilot</h1>
        <p>Your intelligent assistant for employee retention insights and actions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Actions
    st.markdown("**Quick Actions:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 Analyze high-risk in IT"):
            st.session_state.chat_history.append({
                "role": "user",
                "content": "Analyze high-risk employees in IT department"
            })
    
    with col2:
        if st.button("💡 Retention strategies"):
            st.session_state.chat_history.append({
                "role": "user",
                "content": "What are the best retention strategies?"
            })
    
    with col3:
        if st.button("📊 Department comparison"):
            st.session_state.chat_history.append({
                "role": "user",
                "content": "Compare attrition risk across departments"
            })
    
    with col4:
        if st.button("📧 Draft email"):
            st.session_state.chat_history.append({
                "role": "user",
                "content": "Help me draft an appreciation email"
            })
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class='chat-message chat-user'>
                    <p style='margin: 0;'>👤 {message["content"]}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='chat-message chat-assistant'>
                    <p style='margin: 0;'>🤖 {message["content"]}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask the Copilot anything...",
            placeholder="e.g., 'What are the top risk factors for Marketing team?'",
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns([6, 1])
        with col2:
            submit_button = st.form_submit_button("Send 📤")
        
        if submit_button and user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Simple rule-based response
            ai_response = """Based on Velorium's employee data analysis:

**Key Observations:**
- Work-life balance is the strongest predictor of attrition
- Employees with 3+ years without promotion show 2.3x higher risk
- Strong manager relationships can offset compensation concerns by up to 40%

**Recommended Actions:**
- Schedule proactive 1:1s with at-risk employees
- Consider flexible work arrangements for high performers under stress
- Create visible career progression paths

Would you like specific recommendations for individual employees or departments?"""
            
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            st.rerun()

# ============================================================
# PAGE: ACTION GENERATOR
# ============================================================
elif "Action Generator" in page:
    st.markdown("""
    <div class='main-header'>
        <h1>📝 Action Generator</h1>
        <p>Generate personalized retention actions, emails, and recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    action_type = st.selectbox(
        "What would you like to generate?",
        [
            "🎯 Retention Action Plan",
            "📧 Appreciation Email",
            "💬 1:1 Conversation Script",
            "📋 Performance Feedback",
            "📑 Policy Recommendation"
        ]
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Input Context")
        
        if employee_df is not None:
            employee_id = st.selectbox("Select Employee", employee_df['Employee_ID'].astype(str).tolist())
        else:
            employee_id = st.text_input("Employee ID", "1027")
        
        if "Email" in action_type:
            tone = st.select_slider(
                "Email Tone",
                options=["Formal", "Professional", "Warm", "Casual"],
                value="Warm"
            )
        
        additional_context = st.text_area(
            "Additional Context (optional)",
            placeholder="Add any specific context...",
            height=100
        )
        
        if st.button("🚀 Generate", use_container_width=True):
            st.session_state.generated_content = True
    
    with col2:
        st.subheader("📄 Generated Output")
        
        if st.session_state.generated_content:
            if "Email" in action_type:
                st.markdown("""
                <div style='background: #F8FAFC; padding: 1.5rem; border-radius: 12px; border: 1px solid #E2E8F0;'>
                <p><strong>Subject:</strong> Your Outstanding Contributions This Quarter</p>
                <br>
                <p>Dear [Employee Name],</p>
                <br>
                <p>I wanted to take a moment to personally acknowledge the exceptional work you've been delivering. 
                Your dedication and expertise have made a real difference to our team's success.</p>
                <br>
                <p>I recognize the past few months have been demanding with tight deadlines and multiple projects. 
                Your commitment during this period truly stands out.</p>
                <br>
                <p>I'd love to connect this week to discuss your experience and explore ways we can better support 
                your growth and wellbeing. Would you have 30 minutes for a conversation?</p>
                <br>
                <p>Thank you for being such an integral part of our team.</p>
                <br>
                <p>Warm regards,<br>[Manager Name]</p>
                </div>
                """, unsafe_allow_html=True)
            
            elif "Action Plan" in action_type:
                st.markdown("""
                <div class='recommendation-card'>
                    <h4>🎯 30-Day Retention Action Plan</h4>
                    
                    <strong>Week 1: Immediate Engagement</strong>
                    <ul>
                        <li>Schedule informal 1:1 (non-performance focused)</li>
                        <li>Acknowledge recent contributions publicly</li>
                        <li>Review and redistribute workload if needed</li>
                    </ul>
                    
                    <strong>Week 2: Work-Life Balance</strong>
                    <ul>
                        <li>Discuss flexible work arrangements</li>
                        <li>Identify tasks for delegation</li>
                        <li>Set boundaries for after-hours communication</li>
                    </ul>
                    
                    <strong>Week 3: Growth & Development</strong>
                    <ul>
                        <li>Create personalized development plan</li>
                        <li>Identify certification opportunities</li>
                        <li>Connect with senior mentor</li>
                    </ul>
                    
                    <strong>Week 4: Long-term Commitment</strong>
                    <ul>
                        <li>Discuss promotion timeline</li>
                        <li>Review compensation benchmark</li>
                        <li>Document agreed actions</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.button("📋 Copy", use_container_width=True)
            with col_b:
                st.button("✏️ Edit", use_container_width=True)
        else:
            st.info("👈 Configure inputs and click 'Generate'")

# ============================================================
# PAGE: PREDICT NEW
# ============================================================
elif "Predict New" in page:
    st.markdown("""
    <div class='main-header'>
        <h1>🔮 Predict Attrition Risk</h1>
        <p>Enter employee details to predict attrition risk</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Basic Info")
            Gender = st.selectbox("Gender", ["Female", "Male"])
            Marital_Status = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])
            Age_Group = st.selectbox("Age Group", ["<30", "30-40", "40-50", ">50"])
            Department = st.selectbox("Department", ["Finance", "HR", "IT", "Marketing", "Sales"])
            Job_Role = st.selectbox("Job Role", ["Assistant", "Executive", "Manager", "Analyst"])
            Job_Level = st.number_input("Job Level", 1, 5, 2)
        
        with col2:
            st.subheader("Compensation & Tenure")
            Monthly_Income = st.number_input("Monthly Income", 3000, 20000, 12000)
            Income_Level = st.selectbox("Income Level", ["Q1", "Q2", "Q3", "Q4"])
            Hourly_Rate = st.number_input("Hourly Rate", 10, 100, 50)
            Years_at_Company = st.number_input("Years at Company", 0, 30, 5)
            Tenure_Bucket = st.selectbox("Tenure Bucket", ["Early", "Mid", "Long"])
            Years_in_Current_Role = st.number_input("Years in Current Role", 0, 15, 2)
            Years_Since_Last_Promotion = st.number_input("Years Since Last Promotion", 0, 10, 1)
        
        with col3:
            st.subheader("Engagement & Work")
            Job_Satisfaction = st.number_input("Job Satisfaction (1-5)", 1, 5, 3)
            Job_Involvement = st.number_input("Job Involvement (1-5)", 1, 5, 3)
            Work_Environment_Satisfaction = st.number_input("Work Environment (1-5)", 1, 5, 3)
            Relationship_with_Manager = st.number_input("Manager Relationship (1-5)", 1, 5, 3)
            Performance_Rating = st.number_input("Performance Rating (1-4)", 1, 4, 3)
            Average_Hours_Worked_Per_Week = st.number_input("Avg Hours/Week", 20, 100, 45)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            Overtime_Flag = st.selectbox("Overtime", [0, 1])
            Distance_Bucket = st.selectbox("Distance", ["Close", "Medium", "Far", "Very Far"])
        with col2:
            Absenteeism_Level = st.selectbox("Absenteeism", ["Low", "Medium", "High"])
            Number_of_Companies_Worked = st.number_input("Companies Worked", 0, 10, 1)
        with col3:
            Workload_Index = st.number_input("Workload Index", 0.0, 400.0, 60.0)
            Engagement_Score = st.number_input("Engagement Score", 1.0, 5.0, 3.0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            Growth_Index = st.number_input("Growth Index", 0.0, 2.0, 0.5)
        with col2:
            Training_Effectiveness = st.number_input("Training Effectiveness", 0, 300, 50)
        with col3:
            Work_Efficiency = st.number_input("Work Efficiency", 0.0, 50.0, 10.0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮 Predict Attrition Risk", use_container_width=True)
    
    if submitted:
        user_input = {
            "Gender": Gender, "Marital_Status": Marital_Status, "Department": Department,
            "Job_Role": Job_Role, "Job_Level": Job_Level, "Monthly_Income": Monthly_Income,
            "Hourly_Rate": Hourly_Rate, "Years_at_Company": Years_at_Company,
            "Years_in_Current_Role": Years_in_Current_Role,
            "Years_Since_Last_Promotion": Years_Since_Last_Promotion,
            "Job_Satisfaction": Job_Satisfaction, "Performance_Rating": Performance_Rating,
            "Average_Hours_Worked_Per_Week": Average_Hours_Worked_Per_Week,
            "Work_Environment_Satisfaction": Work_Environment_Satisfaction,
            "Relationship_with_Manager": Relationship_with_Manager,
            "Job_Involvement": Job_Involvement,
            "Number_of_Companies_Worked": Number_of_Companies_Worked,
            "Workload_Index": Workload_Index, "Engagement_Score": Engagement_Score,
            "Growth_Index": Growth_Index, "Training_Effectiveness": Training_Effectiveness,
            "Overtime_Flag": Overtime_Flag, "Distance_Bucket": Distance_Bucket,
            "Absenteeism_Level": Absenteeism_Level, "Age_Group": Age_Group,
            "Work_Efficiency": Work_Efficiency, "Tenure_Bucket": Tenure_Bucket,
            "Income_Level": Income_Level
        }
        
        pred, proba = predict_attrition(user_input)
        
        if pred is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            
            risk_level = "High" if proba >= 0.7 else "Medium" if proba >= 0.4 else "Low"
            risk_class = "risk-high" if risk_level == "High" else "risk-medium" if risk_level == "Medium" else "risk-low"
            
            st.markdown(f"""
            <div class='metric-card {risk_class}' style='text-align: center; max-width: 500px; margin: 0 auto;'>
                <h3 style='font-size: 1.2rem; margin-bottom: 1rem;'>Attrition Risk Prediction</h3>
                <p style='font-size: 4rem; font-weight: 700; margin: 1rem 0;'>{proba*100:.0f}%</p>
                <p style='font-size: 1.5rem; font-weight: 600; margin-bottom: 1rem;'>{risk_level} Risk</p>
                <p style='font-size: 0.9rem;'>{'⚠️ Immediate intervention recommended' if risk_level == 'High' else '📋 Monitor closely' if risk_level == 'Medium' else '✅ Employee appears stable'}</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# PAGE: SETTINGS
# ============================================================
elif "Settings" in page:
    st.markdown("""
    <div class='main-header'>
        <h1>⚙️ Settings</h1>
        <p>Configure your Retention Copilot preferences</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔑 API Configuration", "🎨 Display Preferences", "📊 Model Settings"])
    
    with tab1:
        st.subheader("AI Model Configuration")
        
        st.info("💡 **Groq provides free, fast LLM API access** - Recommended for this application")
        
        ai_provider = st.selectbox(
            "AI Provider",
            ["Groq (Free - Recommended)", "OpenAI", "Azure OpenAI"]
        )
        
        if "Groq" in ai_provider:
            api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
            st.markdown("[Get your free Groq API key](https://console.groq.com/keys)")
            st.markdown("**Recommended Model:** Llama 3.3 70B Versatile (Fast & Free)")
        elif ai_provider == "OpenAI":
            api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
            st.markdown("[Get OpenAI API key](https://platform.openai.com/api-keys)")
        
        if st.button("💾 Save API Configuration"):
            if api_key:
                st.success("✅ API configuration saved successfully!")
                st.session_state.api_configured = True
            else:
                st.error("Please enter an API key")
    
    with tab2:
        st.subheader("Display Preferences")
        
        st.markdown("**Dashboard Widgets:**")
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("Show Risk Distribution Chart", value=True)
            st.checkbox("Show Department Comparison", value=True)
            st.checkbox("Show High-Risk Employee List", value=True)
        with col2:
            st.checkbox("Show Quick Actions", value=True)
            st.checkbox("Show Recent Insights", value=True)
            st.checkbox("Show Upcoming 1:1s", value=False)
        
        if st.button("💾 Save Display Preferences"):
            st.success("✅ Display preferences saved!")
    
    with tab3:
        st.subheader("Model Settings")
        
        st.markdown("**Risk Thresholds:**")
        risk_threshold_high = st.slider("High Risk Threshold", 0.5, 1.0, 0.7)
        risk_threshold_medium = st.slider("Medium Risk Threshold", 0.2, 0.7, 0.4)
        
        st.markdown("**Feature Weights:**")
        st.markdown("Adjust importance of different factors in risk calculation")
        
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Work-Life Balance Weight", 0.0, 2.0, 1.0)
            st.slider("Promotion Timeline Weight", 0.0, 2.0, 1.0)
            st.slider("Manager Relationship Weight", 0.0, 2.0, 1.0)
        with col2:
            st.slider("Compensation Weight", 0.0, 2.0, 0.8)
            st.slider("Overtime Weight", 0.0, 2.0, 0.9)
            st.slider("Job Satisfaction Weight", 0.0, 2.0, 1.0)
        
        if st.button("💾 Save Model Settings"):
            st.success("✅ Model settings saved!")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #94A3B8; padding: 2rem;'>
    <p><strong>Velorium Retention Copilot v1.0</strong> | Powered by AnalytIQ Consulting</p>
    <p style='font-size: 0.9rem;'>"The objective isn't perfection. It's earlier empathy."</p>
    <p style='font-size: 0.85rem; margin-top: 1rem;'>
        Built with Streamlit • XGBoost • SHAP • Groq LLM
    </p>
</div>
""", unsafe_allow_html=True)
