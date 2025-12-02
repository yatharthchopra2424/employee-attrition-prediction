"""
⚡ Velorium AI - Premium Employee Retention Platform
"""

import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import json

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "utils"))

@st.cache_data
def load_data():
    try:
        return pd.read_csv("raw_data/employee_attrition_dataset.csv")
    except:
        return None

try:
    from utils.nvidia_chatbot import NVIDIARAGChatbot, generate_email_draft, EMAIL_TEMPLATES
    READY = True
except Exception as e:
    READY = False
    ERR = str(e)

st.set_page_config(page_title="Velorium AI", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

# Premium Dark Theme CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"], [data-testid="stSidebarNav"] { display: none; }

.main { background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%); padding: 0.5rem 1rem; }
.stApp { background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%); }

#MainMenu, footer, header { visibility: hidden; }

/* Premium Header */
.header-bar {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.15) 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 0.75rem 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.75rem;
    backdrop-filter: blur(10px);
}

.logo { 
    font-size: 1.25rem; 
    font-weight: 700; 
    background: linear-gradient(135deg, #6366f1, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.6rem;
    border-radius: 20px;
    font-size: 0.65rem;
    font-weight: 500;
    margin-left: 0.4rem;
}

.status-live { background: rgba(34,197,94,0.2); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
.status-ai { background: rgba(99,102,241,0.2); color: #818cf8; border: 1px solid rgba(99,102,241,0.3); }

/* Chat Container - Full Width */
.chat-box {
    background: rgba(15,15,26,0.6);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 0.75rem;
    height: 58vh;
    overflow-y: auto;
    margin-bottom: 0.5rem;
}

.chat-box::-webkit-scrollbar { width: 4px; }
.chat-box::-webkit-scrollbar-track { background: transparent; }
.chat-box::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 4px; }

/* Messages - Adjusted for full width */
.msg-user {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    padding: 0.5rem 0.75rem;
    border-radius: 12px 12px 4px 12px;
    margin: 0.35rem 0;
    margin-left: 50%;
    font-size: 0.8rem;
    line-height: 1.4;
    max-width: 48%;
}

.msg-ai {
    background: rgba(30,30,50,0.8);
    border: 1px solid rgba(99,102,241,0.2);
    color: #e2e8f0;
    padding: 0.5rem 0.75rem;
    border-radius: 12px 12px 12px 4px;
    margin: 0.35rem 0;
    font-size: 0.8rem;
    line-height: 1.4;
    max-width: 70%;
}

.msg-time { font-size: 0.6rem; opacity: 0.5; margin-top: 0.2rem; }

/* Prediction Card - Compact */
.pred-card {
    border-radius: 10px;
    padding: 0.6rem 0.75rem;
    margin: 0.35rem 0;
    max-width: 60%;
}

.pred-high { background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(220,38,38,0.1)); border: 1px solid rgba(239,68,68,0.4); }
.pred-med { background: linear-gradient(135deg, rgba(245,158,11,0.2), rgba(217,119,6,0.1)); border: 1px solid rgba(245,158,11,0.4); }
.pred-low { background: linear-gradient(135deg, rgba(34,197,94,0.2), rgba(22,163,74,0.1)); border: 1px solid rgba(34,197,94,0.4); }

.pred-title { font-size: 0.7rem; font-weight: 600; color: #94a3b8; margin-bottom: 0.3rem; }
.pred-score { font-size: 1.1rem; font-weight: 700; }
.pred-high .pred-score { color: #ef4444; }
.pred-med .pred-score { color: #f59e0b; }
.pred-low .pred-score { color: #22c55e; }
.pred-factors { font-size: 0.65rem; color: #94a3b8; margin-top: 0.3rem; }

/* Quick Stats Row */
.stats-bar {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}

.stat-item {
    background: rgba(30,30,50,0.5);
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 8px;
    padding: 0.35rem 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.stat-val { font-size: 0.75rem; font-weight: 600; color: #e2e8f0; }
.stat-lbl { font-size: 0.55rem; color: #64748b; }

/* Input Area */
.stTextInput > div > div > input {
    background: rgba(30,30,50,0.8) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
    color: white !important;
    font-size: 0.8rem !important;
    padding: 0.6rem 0.85rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
}

.stTextInput > div > div > input::placeholder { color: #64748b !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    padding: 0.45rem 0.75rem !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.4) !important;
}

/* Form */
[data-testid="stForm"] { background: transparent !important; border: none !important; padding: 0 !important; }

/* Plotly Charts */
.js-plotly-plot { border-radius: 10px; overflow: hidden; }

/* Dialog */
[data-testid="stModal"] > div {
    background: rgba(15,15,26,0.98) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 16px !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(30,30,50,0.6) !important;
    border-radius: 8px !important;
    font-size: 0.75rem !important;
    color: #94a3b8 !important;
}

.streamlit-expanderContent { background: rgba(20,20,35,0.8) !important; }

/* Column gap fix */
[data-testid="column"] { padding: 0 0.25rem !important; }

/* Insights Table Styling */
.insights-table {
    background: rgba(30,30,50,0.6);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    padding: 0.75rem;
    margin: 0.5rem 0;
    max-width: 85%;
}

.insights-table table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.7rem;
}

.insights-table th {
    background: rgba(99,102,241,0.2);
    color: #a5b4fc;
    padding: 0.4rem 0.5rem;
    text-align: left;
    font-weight: 600;
    border-bottom: 1px solid rgba(99,102,241,0.3);
}

.insights-table td {
    padding: 0.4rem 0.5rem;
    color: #e2e8f0;
    border-bottom: 1px solid rgba(99,102,241,0.1);
}

.insights-table tr:hover td {
    background: rgba(99,102,241,0.1);
}

/* Email Draft Card */
.email-draft {
    background: rgba(20,20,35,0.8);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 10px;
    padding: 0.75rem;
    margin: 0.5rem 0;
    max-width: 85%;
    font-size: 0.7rem;
    color: #cbd5e1;
    white-space: pre-wrap;
    line-height: 1.5;
}

.email-draft-header {
    background: linear-gradient(135deg, rgba(139,92,246,0.2), rgba(99,102,241,0.2));
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 0.75rem;
    margin: -0.75rem -0.75rem 0.5rem -0.75rem;
    font-weight: 600;
    color: #a5b4fc;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Risk Gauge */
.risk-gauge-container {
    background: rgba(30,30,50,0.6);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    padding: 0.5rem;
    margin: 0.5rem 0;
    max-width: 300px;
}

/* Action Button Pills */
.action-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin-top: 0.5rem;
}

.action-pill {
    background: rgba(34,197,94,0.15);
    border: 1px solid rgba(34,197,94,0.3);
    color: #22c55e;
    padding: 0.2rem 0.5rem;
    border-radius: 12px;
    font-size: 0.6rem;
    font-weight: 500;
}

/* Summary Card */
.summary-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(139,92,246,0.1));
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    padding: 0.75rem;
    margin: 0.5rem 0;
    max-width: 85%;
}

.summary-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
    font-weight: 600;
    color: #a5b4fc;
    font-size: 0.75rem;
}

.summary-content {
    font-size: 0.7rem;
    color: #cbd5e1;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


def create_chart(df, chart_type):
    colors = ['#6366f1', '#8b5cf6', '#a855f7', '#22c55e', '#f59e0b', '#ef4444']
    
    if chart_type == "dept":
        data = df.groupby('Department')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
        data.columns = ['Department', 'Rate']
        fig = px.bar(data, x='Department', y='Rate', color='Rate', color_continuous_scale='RdYlGn_r')
        fig.update_layout(title="Attrition by Department", height=320)
    elif chart_type == "satisfaction":
        fig = px.histogram(df, x='Job_Satisfaction', color='Attrition', barmode='group',
                          color_discrete_map={'Yes': '#ef4444', 'No': '#22c55e'})
        fig.update_layout(title="Satisfaction Distribution", height=320)
    elif chart_type == "income":
        fig = px.box(df, x='Department', y='Monthly_Income', color='Attrition',
                    color_discrete_map={'Yes': '#ef4444', 'No': '#22c55e'})
        fig.update_layout(title="Income by Department", height=320)
    elif chart_type == "overtime":
        data = df.groupby('Overtime')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
        data.columns = ['Overtime', 'Rate']
        fig = px.pie(data, values='Rate', names='Overtime', hole=0.5, color_discrete_sequence=colors)
        fig.update_layout(title="Overtime Impact", height=320)
    elif chart_type == "tenure":
        fig = px.histogram(df, x='Years_at_Company', color='Attrition', barmode='overlay', opacity=0.7,
                          color_discrete_map={'Yes': '#ef4444', 'No': '#22c55e'})
        fig.update_layout(title="Tenure Distribution", height=320)
    else:
        data = df.groupby('Work_Life_Balance')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
        data.columns = ['WLB', 'Rate']
        fig = px.bar(data, x='WLB', y='Rate', color='Rate', color_continuous_scale='RdYlGn_r')
        fig.update_layout(title="Work-Life Balance Impact", height=320)
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', size=10), margin=dict(l=40, r=20, t=40, b=40),
        showlegend=True, legend=dict(font=dict(size=9))
    )
    return fig


def create_risk_gauge(risk_percentage: float, risk_level: str):
    """Create a visual risk gauge chart."""
    color = '#ef4444' if risk_level == 'HIGH' else '#f59e0b' if risk_level == 'MEDIUM' else '#22c55e'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_percentage,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': '%', 'font': {'size': 24, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#64748b', 'tickfont': {'size': 10}},
            'bar': {'color': color, 'thickness': 0.7},
            'bgcolor': 'rgba(30,30,50,0.6)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': 'rgba(34,197,94,0.2)'},
                {'range': [40, 70], 'color': 'rgba(245,158,11,0.2)'},
                {'range': [70, 100], 'color': 'rgba(239,68,68,0.2)'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.8,
                'value': risk_percentage
            }
        }
    ))
    
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8')
    )
    return fig


def create_employee_metrics_chart(employee: dict):
    """Create a radar chart of employee metrics."""
    categories = ['Satisfaction', 'Work-Life', 'Manager', 'Environment', 'Involvement']
    values = [
        employee.get('Job_Satisfaction', 3),
        employee.get('Work_Life_Balance', 3),
        employee.get('Relationship_with_Manager', 3),
        employee.get('Work_Environment_Satisfaction', 3),
        employee.get('Job_Involvement', 3)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the shape
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(99,102,241,0.3)',
        line=dict(color='#6366f1', width=2),
        name='Current'
    ))
    
    # Add benchmark line
    fig.add_trace(go.Scatterpolar(
        r=[3, 3, 3, 3, 3, 3],
        theta=categories + [categories[0]],
        line=dict(color='rgba(148,163,184,0.5)', width=1, dash='dash'),
        name='Benchmark'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5], tickfont=dict(size=8), gridcolor='rgba(148,163,184,0.2)'),
            angularaxis=dict(tickfont=dict(size=9, color='#94a3b8'), gridcolor='rgba(148,163,184,0.2)'),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        height=200,
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def init():
    if 'bot' not in st.session_state and READY:
        st.session_state.bot = NVIDIARAGChatbot()
    if 'msgs' not in st.session_state:
        st.session_state.msgs = []
    if 'show_email' not in st.session_state:
        st.session_state.show_email = None


def render_insights_table(insights_data: list):
    """Render actionable insights as a styled dataframe."""
    if not insights_data:
        return
    
    # Create DataFrame for proper rendering
    df = pd.DataFrame(insights_data)
    df.columns = ['🚨 Signal', '✅ Recommended Action', '📈 Expected Impact']
    
    # Style the dataframe
    st.markdown("<div style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "🚨 Signal": st.column_config.TextColumn(width="medium"),
            "✅ Recommended Action": st.column_config.TextColumn(width="large"),
            "📈 Expected Impact": st.column_config.TextColumn(width="medium"),
        }
    )
    st.markdown("</div>", unsafe_allow_html=True)


def create_action_gantt_chart(insights_data: list):
    """Create a Gantt chart showing action timeline starting from today."""
    if not insights_data:
        return None
    
    today = datetime.now()
    
    # Define duration mapping for different action types (in days)
    action_durations = {
        "flexible scheduling": {"start_offset": 0, "duration": 14, "phase": "Week 1-2"},
        "overtime reduction": {"start_offset": 3, "duration": 21, "phase": "Week 1-3"},
        "workload redistribution": {"start_offset": 3, "duration": 21, "phase": "Week 1-3"},
        "compensation review": {"start_offset": 7, "duration": 30, "phase": "Week 2-5"},
        "market adjustment": {"start_offset": 14, "duration": 21, "phase": "Week 3-6"},
        "manager coaching": {"start_offset": 0, "duration": 30, "phase": "Week 1-4"},
        "1:1 improvement": {"start_offset": 0, "duration": 14, "phase": "Week 1-2"},
        "role enrichment": {"start_offset": 7, "duration": 28, "phase": "Week 2-6"},
        "project assignment": {"start_offset": 14, "duration": 21, "phase": "Week 3-6"},
        "career pathing": {"start_offset": 7, "duration": 45, "phase": "Week 2-8"},
        "promotion roadmap": {"start_offset": 14, "duration": 60, "phase": "Week 3-12"},
        "proactive": {"start_offset": 0, "duration": 7, "phase": "Week 1"},
        "check-in": {"start_offset": 0, "duration": 7, "phase": "Week 1"},
        "engagement": {"start_offset": 3, "duration": 14, "phase": "Week 1-2"},
    }
    
    gantt_data = []
    colors = ['#6366f1', '#8b5cf6', '#a855f7', '#22c55e', '#f59e0b', '#ec4899']
    
    for i, insight in enumerate(insights_data):
        action = insight.get('Recommended Action', insight.get('action', 'General Action'))
        signal = insight.get('Signal', insight.get('signal', 'Risk Factor'))
        
        # Find matching duration based on action keywords
        start_offset = 0
        duration = 14  # Default 2 weeks
        
        action_lower = action.lower()
        for keyword, timing in action_durations.items():
            if keyword in action_lower:
                start_offset = timing["start_offset"]
                duration = timing["duration"]
                break
        
        start_date = today + timedelta(days=start_offset)
        end_date = start_date + timedelta(days=duration)
        
        gantt_data.append({
            "Task": action[:40] + "..." if len(action) > 40 else action,
            "Signal": signal,
            "Start": start_date,
            "End": end_date,
            "Duration": f"{duration} days",
            "Color": colors[i % len(colors)]
        })
    
    # Sort by start date
    gantt_data.sort(key=lambda x: x["Start"])
    
    # Create Gantt chart using plotly
    fig = go.Figure()
    
    for i, task in enumerate(gantt_data):
        fig.add_trace(go.Bar(
            name=task["Task"],
            y=[task["Task"]],
            x=[(task["End"] - task["Start"]).days],
            base=[(task["Start"] - today).days],
            orientation='h',
            marker=dict(
                color=task["Color"],
                line=dict(color='rgba(255,255,255,0.3)', width=1)
            ),
            hovertemplate=(
                f"<b>{task['Task']}</b><br>"
                f"Signal: {task['Signal']}<br>"
                f"Start: {task['Start'].strftime('%b %d, %Y')}<br>"
                f"End: {task['End'].strftime('%b %d, %Y')}<br>"
                f"Duration: {task['Duration']}<extra></extra>"
            ),
            showlegend=False
        ))
    
    # Add today marker
    fig.add_vline(
        x=0, 
        line_dash="dash", 
        line_color="#22c55e",
        annotation_text="Today",
        annotation_position="top",
        annotation_font_color="#22c55e"
    )
    
    # Calculate max days for x-axis
    max_days = max((t["End"] - today).days for t in gantt_data) + 7
    
    # Generate week markers
    week_markers = list(range(0, max_days + 1, 7))
    week_labels = [f"Week {i}" if i > 0 else "Today" for i in range(len(week_markers))]
    
    fig.update_layout(
        title=dict(
            text="📅 Action Implementation Timeline",
            font=dict(size=14, color='#a5b4fc')
        ),
        xaxis=dict(
            title="Timeline",
            tickmode='array',
            tickvals=week_markers,
            ticktext=week_labels,
            gridcolor='rgba(148,163,184,0.1)',
            tickfont=dict(size=10, color='#94a3b8'),
            range=[-2, max_days + 5]
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=9, color='#e2e8f0'),
            gridcolor='rgba(148,163,184,0.1)'
        ),
        height=max(180, len(gantt_data) * 45 + 80),
        margin=dict(l=10, r=20, t=50, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        bargap=0.3
    )
    
    return fig


def render_email_draft(email_content: str, email_type: str = "support"):
    """Render email draft in a styled card."""
    type_labels = {
        "work_life_support": "💼 Work-Life Support Email",
        "overtime_concern": "⏰ Overtime Concern Email",
        "manager_checkin": "🤝 Manager Check-in Email",
        "appreciation": "⭐ Appreciation Email",
        "career_growth": "📈 Career Growth Email"
    }
    
    label = type_labels.get(email_type, "📧 Draft Email")
    
    st.markdown(f"""
    <div class="email-draft">
        <div class="email-draft-header">
            {label}
        </div>
        {email_content.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)


def render_summary_card(summary: dict):
    """Render risk summary card."""
    risk = summary.get('risk', 'N/A')
    level = summary.get('risk_level', 'UNKNOWN')
    signals = summary.get('signals', 'None identified')
    
    level_color = '#ef4444' if level == 'HIGH' else '#f59e0b' if level == 'MEDIUM' else '#22c55e'
    level_emoji = '🔴' if level == 'HIGH' else '🟡' if level == 'MEDIUM' else '🟢'
    
    st.markdown(f"""
    <div class="summary-card">
        <div class="summary-header">
            {level_emoji} Risk Assessment Summary
        </div>
        <div class="summary-content">
            <strong>Predicted Risk:</strong> <span style="color:{level_color};font-weight:600;">{risk} ({level})</span><br>
            <strong>Key Signals:</strong> {signals}<br>
            <strong>Overall Impact:</strong> Reduced attrition risk via targeted, empathetic interventions
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_pred(pred, emp):
    level = pred.get('risk_level', '').lower()
    css = 'pred-high' if level == 'high' else 'pred-med' if level == 'medium' else 'pred-low'
    factors = pred.get('risk_factors', [])[:4]
    factors_txt = " • ".join(factors) if factors else "No major risks"
    
    st.markdown(f"""
    <div class="pred-card {css}">
        <div class="pred-title">🎯 EMPLOYEE #{pred.get('employee_id')} • {emp.get('Department', 'N/A')} • {emp.get('Job_Role', 'N/A')}</div>
        <div class="pred-score">{pred.get('color')} {pred.get('risk_level')} — {pred.get('risk_percentage')}</div>
        <div class="pred-factors">{factors_txt}</div>
    </div>
    """, unsafe_allow_html=True)


def render_msg(role, txt, time=None):
    css = "msg-user" if role == "user" else "msg-ai"
    t = time or datetime.now().strftime("%H:%M")
    
    # Convert markdown to HTML for proper display
    import re
    # Convert **text** to <strong>text</strong>
    txt = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', txt)
    # Convert *text* to <em>text</em>
    txt = re.sub(r'\*(.+?)\*', r'<em>\1</em>', txt)
    # Convert bullet points
    txt = txt.replace("• ", "● ")
    # Convert newlines to <br>
    txt = txt.replace("\n", "<br>")

    st.markdown(
        f'<div class="{css}">{txt}<div class="msg-time">{t}</div></div>',
        unsafe_allow_html=True
    )


def render_llm_response(txt: str, time=None):
    """Render LLM response with proper markdown formatting using Streamlit native."""
    if not txt:
        return
    
    t = time or datetime.now().strftime("%H:%M")
    
    # Create a styled container for the LLM response
    st.markdown(f"""
    <div style="
        background: rgba(30,30,50,0.8);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        max-width: 90%;
    ">
        <div style="font-size:0.6rem;color:#64748b;margin-bottom:0.5rem;">🤖 AI Recommendations • {t}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Use native streamlit markdown for proper formatting of bullet points and bold text
    st.markdown(txt)


def render_structured_response(structured_output: dict, employee: dict = None):
    """Render the structured response with tables, charts, and email drafts."""
    
    if not structured_output:
        return
    
    # Summary Card
    if 'summary' in structured_output:
        render_summary_card(structured_output['summary'])
    
    # Risk Gauge Chart
    if employee and structured_output.get('summary'):
        risk_val = float(structured_output['summary'].get('risk', '0').replace('%', ''))
        risk_level = structured_output['summary'].get('risk_level', 'UNKNOWN')
        
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.plotly_chart(create_risk_gauge(risk_val, risk_level), use_container_width=True)
        with col2:
            st.plotly_chart(create_employee_metrics_chart(employee), use_container_width=True)
    
    # Insights Table
    if structured_output.get('insights_table'):
        st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
        render_insights_table(structured_output['insights_table'])
        
        # Gantt Chart for Action Timeline
        gantt_fig = create_action_gantt_chart(structured_output['insights_table'])
        if gantt_fig:
            st.plotly_chart(gantt_fig, use_container_width=True)
    
    # Email Drafts with Expanders
    if structured_output.get('email_drafts'):
        st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
        with st.expander("📧 View Personalized Email Drafts", expanded=False):
            for i, draft in enumerate(structured_output['email_drafts']):
                render_email_draft(draft.get('content', ''), draft.get('type', 'appreciation'))
                if i < len(structured_output['email_drafts']) - 1:
                    st.markdown("<hr style='border-color:rgba(99,102,241,0.2);margin:0.5rem 0;'>", unsafe_allow_html=True)



def main():
    init()
    df = load_data()
    
    # Header
    st.markdown("""
    <div class="header-bar">
        <div style="display:flex;align-items:center;">
            <span class="logo">⚡ Velorium AI</span>
            <span class="status-pill status-live">● LIVE</span>
            <span class="status-pill status-ai">🧠 GPT</span>
        </div>
        <div style="color:#64748b;font-size:0.7rem;">Enterprise Retention Platform</div>
    </div>
    """, unsafe_allow_html=True)
    
    if not READY:
        st.error(f"⚠️ {ERR}")
        return
    
    # Quick Stats
    if df is not None:
        total = len(df)
        attrition_rate = (df['Attrition'] == 'Yes').mean() * 100
        avg_sat = df['Job_Satisfaction'].mean()
        ot_rate = (df['Overtime'] == 'Yes').mean() * 100
        
        st.markdown(f"""
        <div class="stats-bar">
            <div class="stat-item"><span class="stat-val">{total:,}</span><span class="stat-lbl">Employees</span></div>
            <div class="stat-item"><span class="stat-val" style="color:#ef4444">{attrition_rate:.1f}%</span><span class="stat-lbl">Attrition</span></div>
            <div class="stat-item"><span class="stat-val" style="color:#22c55e">{avg_sat:.1f}/5</span><span class="stat-lbl">Satisfaction</span></div>
            <div class="stat-item"><span class="stat-val" style="color:#f59e0b">{ot_rate:.0f}%</span><span class="stat-lbl">Overtime</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    # Analytics buttons row
    st.markdown("<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;'>", unsafe_allow_html=True)
    
    charts = [("📊", "Department", "dept"), ("😊", "Satisfaction", "satisfaction"), ("💰", "Income", "income"),
              ("⏰", "Overtime", "overtime"), ("📅", "Tenure", "tenure"), ("⚖️", "Work-Life", "worklife")]
    
    btn_cols = st.columns([1, 1, 1, 1, 1, 1, 4])
    for i, (icon, label, key) in enumerate(charts):
        with btn_cols[i]:
            if st.button(f"{icon}", key=f"c_{key}", help=label):
                st.session_state[f"show_{key}"] = True
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Show chart dialogs
    if df is not None:
        for icon, label, key in charts:
            if st.session_state.get(f"show_{key}"):
                @st.dialog(f"{icon} {label} Analysis", width="large")
                def show(k=key, l=label):
                    st.plotly_chart(create_chart(df, k), use_container_width=True)
                    # Quick insight for this chart
                    if k == "dept":
                        high = df.groupby('Department')['Attrition'].apply(lambda x: (x == 'Yes').mean()).idxmax()
                        st.caption(f"🚨 Highest attrition: **{high}**")
                    elif k == "overtime":
                        ot = (df[df['Overtime']=='Yes']['Attrition']=='Yes').mean()*100
                        no_ot = (df[df['Overtime']=='No']['Attrition']=='Yes').mean()*100
                        st.caption(f"⚠️ With OT: **{ot:.0f}%** vs Without: **{no_ot:.0f}%**")
                    if st.button("Close", key=f"cls_{k}"):
                        st.session_state[f"show_{k}"] = False
                        st.rerun()
                show()
                st.session_state[f"show_{key}"] = False
    
    # Full width Chat Area
    if not st.session_state.msgs:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#64748b;">
            <div style="font-size:2.5rem;margin-bottom:0.5rem;">💬</div>
            <div style="font-size:0.85rem;font-weight:500;">Ask about any employee</div>
            <div style="font-size:0.7rem;margin-top:0.3rem;color:#475569;">Try: "Analyze employee 232" or "Show high risk employees"</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for m in st.session_state.msgs:
            if m['type'] == 'pred':
                render_pred(m['pred'], m.get('emp', {}))
            elif m['type'] == 'structured':
                render_structured_response(m.get('structured'), m.get('emp'))
            elif m['type'] == 'llm':
                # Use native markdown for LLM responses
                render_llm_response(m['txt'], m.get('time'))
            else:
                render_msg(m['role'], m['txt'], m.get('time'))
    
    # Input - Full width
    with st.form("chat", clear_on_submit=True):
        c1, c2, c3 = st.columns([8, 1, 1])
        with c1:
            inp = st.text_input("msg", placeholder="Message Velorium AI...", label_visibility="collapsed")
        with c2:
            send = st.form_submit_button("Send")
        with c3:
            if st.form_submit_button("🗑️"):
                st.session_state.msgs = []
                st.session_state.bot.clear_history()
                st.rerun()
    
    if send and inp:
        now = datetime.now().strftime("%H:%M")
        st.session_state.msgs.append({'type': 'msg', 'role': 'user', 'txt': inp, 'time': now})
        
        with st.spinner(""):
            result = st.session_state.bot.process_message(inp)
            
            # Handle 4-tuple return (prediction, employee, structured_output, llm_response)
            if len(result) == 4:
                pred, emp, structured_output, llm_resp = result
            else:
                # Fallback for old 3-tuple
                pred, emp, resp = result
                structured_output = None
                llm_resp = resp
            
            if pred and pred.get('success'):
                st.session_state.msgs.append({'type': 'pred', 'pred': pred, 'emp': emp, 'time': now})
            
            if structured_output:
                st.session_state.msgs.append({
                    'type': 'structured', 
                    'structured': structured_output, 
                    'emp': emp, 
                    'time': datetime.now().strftime("%H:%M")
                })
            
            if llm_resp:
                st.session_state.msgs.append({
                    'type': 'llm',  # Use special type for LLM responses
                    'role': 'ai', 
                    'txt': llm_resp, 
                    'time': datetime.now().strftime("%H:%M")
                })
        st.rerun()


if __name__ == "__main__":
    main()
