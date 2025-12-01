# Quick Start Guide for Velorium Retention Copilot

## 🚀 Running the Application

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure API Key (Optional - for AI features)

**Option A: Using Groq (Free & Recommended)**

1. Get API key: https://console.groq.com/keys
2. Create `.env` file in project root:
   ```
   GROQ_API_KEY=your_key_here
   ```

**Option B: Skip AI Features**
- The app works without API keys
- AI chat and recommendations will show template responses

### Step 3: Run the Application
```bash
streamlit run app_mvp.py
```

### Step 4: Access the App
Open browser: http://localhost:8501

## 📱 Features Overview

### 📊 Dashboard
- View organization-wide risk metrics
- See department comparisons
- Identify high-risk employees

### 👥 Employee Analysis
- Select any employee ID
- View detailed risk assessment
- Get SHAP-based explanations
- Receive AI recommendations

### 💬 AI Copilot Chat
- Ask retention-related questions
- Get insights about departments
- Generate action recommendations

### 📝 Action Generator
- Create retention action plans
- Draft appreciation emails
- Generate 1:1 conversation scripts
- Build policy recommendations

### 🔮 Predict New Employee
- Enter employee details
- Get instant risk prediction
- Understand risk factors
- Receive intervention suggestions

## 🎯 Sample Workflows

### Workflow 1: Identify & Intervene
1. Go to Dashboard → View high-risk employees
2. Click "View Details" on Employee #1027
3. Review risk factors and recommendations
4. Generate appreciation email in Action Generator
5. Schedule 1:1 using conversation script

### Workflow 2: Department Analysis
1. Open AI Copilot Chat
2. Ask: "Analyze high-risk employees in IT department"
3. Review insights and recommendations
4. Generate department-wide policy suggestions

### Workflow 3: Predict New Hire Risk
1. Go to "Predict New" page
2. Enter employee details
3. View risk assessment
4. Get proactive retention strategies

## ⚙️ Troubleshooting

**Model file not found:**
- Ensure `models/attrition_model.pkl` exists
- Run the Jupyter notebook to train the model

**Data file not found:**
- Ensure `raw_data/employee_attrition_dataset.csv` exists
- App will work in demo mode without data

**AI features not working:**
- Check `.env` file has valid API key
- Verify API key format (gsk_ for Groq, sk- for OpenAI)
- Check internet connection

**Dependencies error:**
- Update pip: `pip install --upgrade pip`
- Try: `pip install -r requirements.txt --no-cache-dir`

## 💡 Tips for Best Experience

1. **Start with Dashboard** - Get organizational overview
2. **Use Employee Analysis** - Deep dive into individuals
3. **Chat for Quick Insights** - Ask natural language questions
4. **Action Generator** - Create ready-to-use content
5. **Configure API Key** - Unlock full AI capabilities

## 🎓 Understanding the Output

### Risk Score Interpretation
- **High (70-100%)**: Immediate intervention needed
- **Medium (40-69%)**: Monitor closely, proactive engagement
- **Low (0-39%)**: Stable, maintain positive relationship

### SHAP Values
- **Positive SHAP** = Increases attrition risk
- **Negative SHAP** = Decreases attrition risk
- **Magnitude** = Strength of impact

### Recommendations
- **Immediate** = This week
- **Short-term** = Next 30 days
- **Long-term** = Next quarter

---

**Need Help?** Open an issue on GitHub or refer to the full README.md
