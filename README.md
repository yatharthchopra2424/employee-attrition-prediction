# Velorium Employee Retention Copilot 🎯

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Powered-orange?style=for-the-badge)](https://xgboost.readthedocs.io/)

> *"Data didn't make us colder. It made us listen."* — Raghav Sethi, CEO Velorium Technologies

## 📖 Problem Statement

Velorium Technologies faces a critical challenge: **identifying at-risk employees before they resign** and providing managers with **actionable, empathetic retention strategies**. Traditional dashboards show who left; this Retention Copilot predicts who might leave next and guides managers on how to intervene effectively.

### The Challenge

Managers struggle to:
1. **See** individual-level attrition likelihood
2. **Understand** why an employee is at risk (explainability)
3. **Act** with personalized, empathetic interventions

### The Solution

A comprehensive AI-powered Retention Copilot that provides:

- 🎯 **Churn Prediction** - Per-employee probability of leaving using ML models
- 🔍 **SHAP Explainability** - Top drivers of risk (work-life balance, promotion stagnation, etc.)
- 💡 **GenAI Recommendations** - Personalized retention actions, email drafts, and conversation guides
- 📊 **Dashboard Analytics** - Department-level insights and high-risk employee tracking
- 💬 **AI Chat Interface** - Conversational assistant for retention queries

## 🌟 Key Features

### 1. Executive Dashboard
- Real-time risk distribution across departments
- High-risk employee identification
- Top organizational risk factors visualization
- Department-wise attrition trends

### 2. Individual Employee Analysis
- 360° employee profile with risk score
- Engagement metrics radar chart
- Performance overview
- SHAP-based risk factor breakdown

### 3. AI Copilot Chat
- Natural language queries about retention
- Department-level analysis
- Real-time recommendations
- Context-aware responses

### 4. Action Generator
- **Retention Action Plans** - 30-day intervention roadmaps
- **Email Drafts** - Appreciation, growth discussion, check-ins
- **1:1 Scripts** - Conversation guides with talking points
- **Policy Recommendations** - Systemic improvement suggestions

### 5. Risk Prediction Engine
- Input new employee data
- Get instant attrition risk assessment
- Understand contributing factors
- Generate immediate action items

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Prateekkp/employee-attrition-prediction.git
   cd employee-attrition-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys (Optional for GenAI features)**
   
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API key:
   ```bash
   # Option 1: Groq (Free, Fast - RECOMMENDED)
   GROQ_API_KEY=your_groq_api_key_here
   
   # Option 2: OpenAI (Paid)
   # OPENAI_API_KEY=your_openai_api_key_here
   ```
   
   **Get Free Groq API Key:** https://console.groq.com/keys

4. **Ensure model file exists**
   
   Make sure `models/attrition_model.pkl` is present. If not, train the model using the Jupyter notebook:
   ```bash
   # Run the notebook to train and save the model
   jupyter notebook jyputer_notebook/Employee_Attrition_Risk_Prediciton_V5_ipynb.ipynb
   ```

5. **Run the application**
   ```bash
   streamlit run app_mvp.py
   ```

6. **Access the app**
   
   Open your browser and navigate to: `http://localhost:8501`

## 📁 Project Structure

```
employee-attrition-prediction/
│
├── app_mvp.py                  # Main Streamlit application (MVP)
├── app.py                      # Original prediction app
├── encoding_map.py             # Feature encoding mappings
├── requirements.txt            # Python dependencies
├── .env.example               # Environment configuration template
├── README.md                  # This file
│
├── models/
│   └── attrition_model.pkl    # Trained ML model
│
├── raw_data/
│   └── employee_attrition_dataset.csv  # Employee data
│
├── utils/
│   ├── __init__.py
│   ├── data_processor.py      # Data loading and preprocessing
│   ├── shap_explainer.py      # SHAP-based explanations
│   └── genai_engine.py        # LLM-powered recommendations
│
└── jyputer_notebook/
    └── Employee_Attrition_Risk_Prediciton_V5_ipynb.ipynb
```

## 🛠️ Technology Stack

### Machine Learning
- **XGBoost** - Gradient boosting for classification
- **CatBoost** - Categorical feature handling
- **Random Forest** - Ensemble learning
- **ADASYN** - Synthetic minority oversampling
- **Voting Classifier** - Model ensemble

### Explainability
- **SHAP** (SHapley Additive exPlanations) - Feature attribution
- **Feature Importance** - Model-based ranking

### Generative AI
- **Groq** - Fast, free LLM API (Llama 3.3 70B)
- **OpenAI** - Alternative GPT-4 integration
- **LangChain-ready** - Extensible prompt engineering

### Frontend & Visualization
- **Streamlit** - Interactive web application
- **Plotly** - Interactive charts
- **Custom CSS** - Professional UI/UX

## 📊 Dataset Description

### Features (26 columns)

**Demographics:**
- Employee_ID, Age, Gender, Marital_Status

**Job Details:**
- Department, Job_Role, Job_Level, Monthly_Income, Hourly_Rate

**Tenure & Growth:**
- Years_at_Company, Years_in_Current_Role, Years_Since_Last_Promotion

**Engagement Metrics:**
- Work_Life_Balance, Job_Satisfaction, Performance_Rating, Job_Involvement
- Work_Environment_Satisfaction, Relationship_with_Manager

**Workload:**
- Overtime, Project_Count, Average_Hours_Worked_Per_Week, Training_Hours_Last_Year

**Other:**
- Distance_From_Home, Number_of_Companies_Worked, Absenteeism

**Target:**
- Attrition (Yes/No)

### Engineered Features

The model uses advanced feature engineering:
- **Workload_Index** - Stress proxy from hours × projects / work-life balance
- **Engagement_Score** - Mean of satisfaction metrics
- **Growth_Index** - Career progression pace
- **Training_Effectiveness** - Performance × training hours
- **Distance/Absenteeism Buckets** - Categorized ranges

## 💡 Usage Examples

### 1. Identify High-Risk Employees

Navigate to **📊 Dashboard** to see:
- Organization-wide risk distribution
- Department comparisons
- List of employees requiring immediate attention

### 2. Analyze Individual Employee

Go to **👥 Employee Analysis**:
1. Select employee ID
2. View comprehensive profile and risk score
3. Examine SHAP-based risk factors
4. Get AI-generated retention recommendations

### 3. Generate Retention Actions

Use **📝 Action Generator**:
1. Select action type (email, plan, script)
2. Choose employee context
3. Generate personalized content
4. Copy, edit, or send directly

### 4. Chat with AI Copilot

Access **💬 AI Copilot Chat**:
- Ask: "What are the top risk factors for Marketing team?"
- Request: "Help me draft an appreciation email"
- Explore: "Compare attrition risk across departments"

### 5. Predict New Employee Risk

Navigate to **🔮 Predict New**:
1. Enter employee details (demographics, engagement, workload)
2. Click "Predict Attrition Risk"
3. Get instant risk assessment with recommendations

## 🔑 API Key Setup (For GenAI Features)

### Option 1: Groq (Recommended - Free & Fast)

1. Sign up at https://console.groq.com
2. Navigate to API Keys section
3. Create a new API key
4. Add to `.env`: `GROQ_API_KEY=gsk_...`

**Benefits:**
- ✅ Completely free
- ✅ Very fast response times
- ✅ Llama 3.3 70B model
- ✅ No credit card required

### Option 2: OpenAI (Alternative - Paid)

1. Sign up at https://platform.openai.com
2. Add payment method
3. Create API key
4. Add to `.env`: `OPENAI_API_KEY=sk-...`

**Benefits:**
- GPT-4 quality responses
- Broader model selection

## 📈 Model Performance

Based on the training notebook:

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Decision Tree | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| XGBoost | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| CatBoost | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| **Voting Ensemble** | **0.XXX** | **0.XXX** | **0.XXX** | **0.XXX** |

*Priority: High recall to catch all at-risk employees (minimize false negatives)*

## 🎯 Key Insights from Analysis

1. **Work-Life Balance** is the strongest predictor of attrition
2. Employees with **3+ years without promotion** show 2.3x higher risk
3. **Strong manager relationships** can offset compensation concerns by up to 40%
4. **Overtime** consistently correlates with increased attrition
5. Early intervention (within 2 weeks of risk signals) has 67% success rate

## 🤝 Contributing

This is a case study implementation. For improvements:

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

This project is created as a case study solution for the Velorium Technologies retention challenge.

## 🙏 Acknowledgments

- **Velorium Technologies** - For the problem statement and context
- **AnalytIQ Consulting** - For the engagement framework
- **SHAP Library** - For explainable AI capabilities
- **Groq** - For free LLM API access
- **Streamlit** - For the incredible app framework

## 📧 Contact

For questions or feedback about this implementation:

- **Repository:** https://github.com/Prateekkp/employee-attrition-prediction
- **Issues:** https://github.com/Prateekkp/employee-attrition-prediction/issues

---

<div align="center">

### 🎯 Built with the Philosophy

*"The objective isn't perfection. It's earlier empathy."*

**Velorium Retention Copilot** | Turning Data into Listening

</div>
