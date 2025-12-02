"""
⚡ Velorium AI - Premium Employee Retention Platform
"""

import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "utils"))

@st.cache_data
def load_data():
    try:
        return pd.read_csv("raw_data/employee_attrition_dataset.csv")
    except:
        return None

try:
    from utils.nvidia_chatbot import NVIDIARAGChatbot
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


def init():
    if 'bot' not in st.session_state and READY:
        st.session_state.bot = NVIDIARAGChatbot()
    if 'msgs' not in st.session_state:
        st.session_state.msgs = []


def render_pred(pred, emp):
    level = pred.get('risk_level', '').lower()
    css = 'pred-high' if level == 'high' else 'pred-med' if level == 'medium' else 'pred-low'
    factors = pred.get('risk_factors', [])[:3]
    factors_txt = " • ".join(factors) if factors else "No major risks"
    
    st.markdown(f"""
    <div class="pred-card {css}">
        <div class="pred-title">🎯 EMPLOYEE #{pred.get('employee_id')} • {emp.get('Department', 'N/A')}</div>
        <div class="pred-score">{pred.get('color')} {pred.get('risk_level')} — {pred.get('risk_percentage')}</div>
        <div class="pred-factors">{factors_txt}</div>
    </div>
    """, unsafe_allow_html=True)


def render_msg(role, txt, time=None):
    css = "msg-user" if role == "user" else "msg-ai"
    t = time or datetime.now().strftime("%H:%M")
    txt = txt.replace("\n", "<br>")

    st.markdown(
        f'<div class="{css}">{txt}<div class="msg-time">{t}</div></div>',
        unsafe_allow_html=True
    )



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
            pred, emp, resp = st.session_state.bot.process_message(inp)
            if pred and pred.get('success'):
                st.session_state.msgs.append({'type': 'pred', 'pred': pred, 'emp': emp, 'time': now})
            st.session_state.msgs.append({'type': 'msg', 'role': 'ai', 'txt': resp, 'time': datetime.now().strftime("%H:%M")})
        st.rerun()


if __name__ == "__main__":
    main()
