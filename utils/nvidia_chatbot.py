# ================================
#  NVIDIA RAG Chatbot (Updated)
#  Structured concise retention output with actionable insights
# ================================

import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from utils.encoding_map import ENCODING_MAP
except ImportError:
    ENCODING_MAP = {}

NVIDIA_API_KEY = "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"

MODEL_FEATURES = [
    'Gender', 'Marital_Status', 'Department', 'Job_Role', 'Job_Level',
    'Monthly_Income', 'Hourly_Rate', 'Years_at_Company', 'Years_in_Current_Role',
    'Years_Since_Last_Promotion', 'Job_Satisfaction', 'Performance_Rating',
    'Average_Hours_Worked_Per_Week', 'Work_Environment_Satisfaction',
    'Relationship_with_Manager', 'Job_Involvement', 'Number_of_Companies_Worked',
    'Workload_Index', 'Engagement_Score', 'Growth_Index', 'Training_Effectiveness',
    'Overtime_Flag', 'Distance_Bucket', 'Absenteeism_Level', 'Age_Group',
    'Work_Efficiency', 'Tenure_Bucket', 'Income_Level'
]

# ==========================================================
# 🔥 Actionable Insights Generator
# ==========================================================

ACTION_MAP = {
    "poor work-life balance": {
        "action": "Implement flexible scheduling",
        "impact": "30-40% improvement in work-life satisfaction",
        "email_template": "work_life_support"
    },
    "frequent overtime": {
        "action": "Overtime reduction & workload redistribution",
        "impact": "25% reduction in burnout indicators",
        "email_template": "overtime_concern"
    },
    "below-average compensation": {
        "action": "Compensation review & market adjustment",
        "impact": "15-20% reduction in compensation-driven attrition",
        "email_template": "growth_opportunity"
    },
    "weak manager relationship": {
        "action": "Manager coaching & 1:1 improvement plan",
        "impact": "35% improvement in engagement scores",
        "email_template": "manager_checkin"
    },
    "low job satisfaction": {
        "action": "Role enrichment & meaningful project assignment",
        "impact": "25% boost in job satisfaction metrics",
        "email_template": "appreciation"
    },
    "stagnant career growth": {
        "action": "Career pathing discussion & promotion roadmap",
        "impact": "40% reduction in growth-driven departures",
        "email_template": "career_growth"
    }
}

EMAIL_TEMPLATES = {
    "work_life_support": """Subject: Your Well-being Matters to Us

Hi {name},

I've been reflecting on the demands we've placed on the team lately, and I want to personally check in with you.

Your dedication hasn't gone unnoticed, but I also want to ensure we're supporting your well-being. I'd like to discuss some flexible arrangements that might help:
• Adjusted work hours that better fit your personal schedule
• Remote work options for better work-life integration
• Workload review to ensure sustainable expectations

Could we schedule 30 minutes this week to discuss what would work best for you?

Your health and happiness are priorities for us.

Warm regards,
{manager}""",

    "overtime_concern": """Subject: Let's Talk About Your Workload

Hi {name},

I've noticed the extra hours you've been putting in, and while I deeply appreciate your commitment, I'm concerned about the sustainability of this pace.

You're a valued member of our team, and I want to make sure we're not burning you out. Let's discuss:
• How we can better distribute the workload
• Additional resources or support you might need
• Realistic timelines for current projects

Your long-term success here matters more than any short-term deadline.

Let's connect this week?

Best,
{manager}""",

    "manager_checkin": """Subject: I Value Our Working Relationship

Hi {name},

I wanted to reach out personally because your perspective matters to me.

I'm always looking to improve how I can support you better. Would you be open to sharing:
• What's working well in our collaboration
• Areas where I could provide better support
• Any concerns or ideas you'd like to discuss

I'm committed to being the kind of manager who helps you thrive. Let's have an open conversation.

Looking forward to hearing from you,
{manager}""",

    "appreciation": """Subject: Thank You for Your Outstanding Contributions

Hi {name},

I wanted to take a moment to recognize the exceptional work you've been doing.

Your contributions to {department} haven't gone unnoticed:
• Your expertise and dedication are invaluable
• The team benefits greatly from your presence
• Your work directly impacts our success

You matter here, and I want you to know that we're invested in your continued growth and happiness.

Let's schedule time to discuss your aspirations and how we can support them.

With gratitude,
{manager}""",

    "career_growth": """Subject: Let's Map Your Career Journey Together

Hi {name},

I've been thinking about your career development and wanted to have a dedicated conversation about your future here.

After {tenure} years of valuable contributions, I believe you're ready for new challenges. I'd like to discuss:
• Your career aspirations and goals
• Skills you're excited to develop
• Leadership opportunities that align with your interests
• A clear roadmap for your next career milestone

You have tremendous potential, and I want to ensure we're nurturing it.

Can we block 45 minutes this week?

Best,
{manager}"""
}


def generate_actionable_insights(pred: dict, employee: dict) -> dict:
    """Generate comprehensive actionable insights with recommendations."""
    
    signals = pred.get("risk_factors", [])
    insights = []
    
    for signal in signals:
        signal_lower = signal.lower()
        for key, action_data in ACTION_MAP.items():
            if key in signal_lower:
                insights.append({
                    "signal": signal,
                    "action": action_data["action"],
                    "impact": action_data["impact"],
                    "email_template": action_data["email_template"]
                })
                break
    
    # Add default insights if none found
    if not insights:
        insights.append({
            "signal": "General retention risk",
            "action": "Proactive manager check-in & engagement boost",
            "impact": "20% improvement in retention likelihood",
            "email_template": "appreciation"
        })
    
    return {
        "risk_level": pred.get("risk_level", "UNKNOWN"),
        "risk_percentage": pred.get("risk_percentage", "N/A"),
        "insights": insights,
        "employee_context": {
            "name": f"Employee #{employee.get('Employee_ID', 'N/A')}",
            "department": employee.get("Department", "N/A"),
            "role": employee.get("Job_Role", "N/A"),
            "tenure": employee.get("Years_at_Company", "N/A"),
            "satisfaction": employee.get("Job_Satisfaction", "N/A"),
            "work_life_balance": employee.get("Work_Life_Balance", "N/A"),
            "manager_relationship": employee.get("Relationship_with_Manager", "N/A")
        }
    }


def generate_email_draft(template_key: str, employee: dict, manager_name: str = "Your Manager") -> str:
    """Generate personalized email draft based on template and employee context."""
    
    template = EMAIL_TEMPLATES.get(template_key, EMAIL_TEMPLATES["appreciation"])
    
    return template.format(
        name=f"Employee #{employee.get('Employee_ID', 'Team Member')}",
        manager=manager_name,
        department=employee.get("Department", "our team"),
        tenure=employee.get("Years_at_Company", "your")
    )


def format_retention_output(pred: dict, employee: dict = None, include_drafts: bool = True) -> dict:
    """Return comprehensive structured output with actionable insights."""
    
    risk = pred.get("risk_percentage", "N/A")
    signals = pred.get("risk_factors", [])[:4]
    signals_txt = ", ".join(s.replace("⚠️ ", "") for s in signals) or "None identified"
    
    # Generate insights table data
    insights_data = []
    email_drafts = []
    
    for signal in signals:
        signal_lower = signal.lower()
        matched = False
        for key, action_data in ACTION_MAP.items():
            if key in signal_lower:
                insights_data.append({
                    "Signal": signal,
                    "Recommended Action": action_data["action"],
                    "Expected Impact": action_data["impact"]
                })
                if include_drafts and employee:
                    email_drafts.append({
                        "type": action_data["email_template"],
                        "content": generate_email_draft(action_data["email_template"], employee)
                    })
                matched = True
                break
        
        if not matched:
            insights_data.append({
                "Signal": signal,
                "Recommended Action": "Manager 1:1 & engagement review",
                "Expected Impact": "Improved retention likelihood"
            })
    
    # Add default if empty
    if not insights_data:
        insights_data.append({
            "Signal": "General risk indicators",
            "Recommended Action": "Proactive retention conversation",
            "Expected Impact": "15-25% risk reduction"
        })
        if include_drafts and employee:
            email_drafts.append({
                "type": "appreciation",
                "content": generate_email_draft("appreciation", employee)
            })
    
    return {
        "summary": {
            "risk": risk,
            "risk_level": pred.get("risk_level", "UNKNOWN"),
            "signals": signals_txt
        },
        "insights_table": insights_data,
        "email_drafts": email_drafts,
        "overall_impact": "Reduced attrition risk via targeted, empathetic interventions"
    }


# ==========================================================
# FAISS Vector DB with Enhanced Context Retrieval
# ==========================================================

class EmployeeVectorDB:
    def __init__(self, csv_path="raw_data/employee_attrition_dataset.csv"):
        self.csv_path = csv_path
        self.df = None
        self.index = None
        self.embeddings = None
        self.embedding_dim = 256
        self._load_data()
        if FAISS_AVAILABLE and self.df is not None:
            self._build_index()

    def _load_data(self):
        try:
            if os.path.exists(self.csv_path):
                self.df = pd.read_csv(self.csv_path)
        except Exception as e:
            print(f"Error loading data: {e}")

    def _create_text_representation(self, row):
        """Enhanced text representation with more context for better RAG retrieval."""
        return (
            f"Employee {row.get('Employee_ID')} Age {row.get('Age')} "
            f"{row.get('Gender')} {row.get('Marital_Status')} "
            f"Department {row.get('Department')} Role {row.get('Job_Role')} "
            f"Level {row.get('Job_Level')} Income {row.get('Monthly_Income')} "
            f"Years {row.get('Years_at_Company')} YearsInRole {row.get('Years_in_Current_Role')} "
            f"YearsSincePromotion {row.get('Years_Since_Last_Promotion')} "
            f"Satisfaction {row.get('Job_Satisfaction')} Performance {row.get('Performance_Rating')} "
            f"WorkLife {row.get('Work_Life_Balance')} Environment {row.get('Work_Environment_Satisfaction')} "
            f"Overtime {row.get('Overtime')} AvgHours {row.get('Average_Hours_Worked_Per_Week')} "
            f"Manager {row.get('Relationship_with_Manager')} Involvement {row.get('Job_Involvement')} "
            f"Companies {row.get('Number_of_Companies_Worked')} Distance {row.get('Distance_from_Home')} "
            f"Training {row.get('Training_Hours_Last_Year')} Attrition {row.get('Attrition')}"
        )

    def _simple_embedding(self, text: str):
        emb = np.zeros(self.embedding_dim, dtype=np.float32)
        t = text.lower()
        for i in range(len(t) - 2):
            trigram = t[i:i+3]
            emb[hash(trigram) % self.embedding_dim] += 1
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def _build_index(self):
        embeddings = []
        for _, row in self.df.iterrows():
            embeddings.append(self._simple_embedding(self._create_text_representation(row)))
        self.embeddings = np.array(embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(self.embeddings)

    def get_employee_by_id(self, emp_id: int):
        row = self.df[self.df["Employee_ID"] == emp_id]
        return row.iloc[0].to_dict() if not row.empty else None

    def get_similar_employees(self, emp_id: int, k: int = 5) -> List[dict]:
        """Retrieve similar employees for comparative context."""
        if self.index is None:
            return []
        
        emp = self.get_employee_by_id(emp_id)
        if not emp:
            return []
        
        query_emb = self._simple_embedding(self._create_text_representation(emp))
        query_emb = np.array([query_emb], dtype=np.float32)
        
        distances, indices = self.index.search(query_emb, k + 1)
        
        similar = []
        for idx in indices[0][1:]:  # Skip first (self)
            if 0 <= idx < len(self.df):
                similar.append(self.df.iloc[idx].to_dict())
        
        return similar

    def get_department_context(self, department: str) -> dict:
        """Get department-level statistics for context."""
        if self.df is None:
            return {}
        
        dept_df = self.df[self.df["Department"] == department]
        if dept_df.empty:
            return {}
        
        return {
            "total_employees": len(dept_df),
            "avg_satisfaction": dept_df["Job_Satisfaction"].mean(),
            "avg_tenure": dept_df["Years_at_Company"].mean(),
            "attrition_rate": (dept_df["Attrition"] == "Yes").mean() * 100,
            "avg_income": dept_df["Monthly_Income"].mean(),
            "overtime_rate": (dept_df["Overtime"] == "Yes").mean() * 100
        }

    def get_enhanced_context(self, emp_id: int) -> dict:
        """Get comprehensive context for an employee including similar profiles and department stats."""
        emp = self.get_employee_by_id(emp_id)
        if not emp:
            return {}
        
        similar = self.get_similar_employees(emp_id, k=3)
        dept_context = self.get_department_context(emp.get("Department", ""))
        
        # Calculate comparison metrics
        similar_attrition = sum(1 for s in similar if s.get("Attrition") == "Yes") / max(len(similar), 1) * 100
        
        return {
            "employee": emp,
            "similar_profiles": similar,
            "department_stats": dept_context,
            "similar_attrition_rate": similar_attrition,
            "context_summary": (
                f"Similar employees ({len(similar)}) have {similar_attrition:.0f}% attrition rate. "
                f"Department avg satisfaction: {dept_context.get('avg_satisfaction', 0):.1f}/5"
            )
        }


# ==========================================================
# ML Predictor with Enhanced Risk Detection
# ==========================================================

class AttritionPredictor:
    def __init__(self, path="models/attrition_model.pkl"):
        self.model = None
        self.columns = MODEL_FEATURES
        self._load_model(path)

    def _load_model(self, path):
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    self.model = data["model"]
                    self.columns = data["columns"]
                else:
                    self.model = data
        except Exception as e:
            print("Model load error:", e)

    def predict(self, row: dict):
        try:
            # --- Build feature vector ---
            features = []
            for feat in MODEL_FEATURES:
                val = row.get(feat, 0)
                val = 0 if isinstance(val, str) else val
                features.append(float(val))

            X = np.array([features])
            if hasattr(self.model, "predict_proba"):
                p = self.model.predict_proba(X)[0][1]
            else:
                p = float(self.model.predict(X)[0])

            level = "HIGH" if p >= 0.7 else "MEDIUM" if p >= 0.4 else "LOW"
            color = "🔴" if p >= 0.7 else "🟡" if p >= 0.4 else "🟢"

            # Enhanced risk factors detection
            f = []
            if row.get("Work_Life_Balance", 5) <= 2: 
                f.append("poor work-life balance")
            if row.get("Overtime") == "Yes" or row.get("Average_Hours_Worked_Per_Week", 40) > 50: 
                f.append("frequent overtime")
            if row.get("Monthly_Income", 10000) < 6000: 
                f.append("below-average compensation")
            if row.get("Relationship_with_Manager", 5) <= 2: 
                f.append("weak manager relationship")
            if row.get("Job_Satisfaction", 5) <= 2:
                f.append("low job satisfaction")
            if row.get("Years_Since_Last_Promotion", 0) >= 4:
                f.append("stagnant career growth")
            if row.get("Work_Environment_Satisfaction", 5) <= 2:
                f.append("poor work environment")
            if row.get("Job_Involvement", 5) <= 2:
                f.append("low job involvement")

            return {
                "success": True,
                "employee_id": row.get("Employee_ID"),
                "risk_probability": float(p),
                "risk_percentage": f"{p*100:.1f}%",
                "risk_level": level,
                "color": color,
                "risk_factors": f[:4]  # Return up to 4 factors
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "risk_level": "ERROR",
                "risk_percentage": "N/A",
                "risk_factors": []
            }


# ==========================================================
# 🤖 NVIDIARAGChatbot — Enhanced with Actionable Insights
# ==========================================================

class NVIDIARAGChatbot:

    def __init__(self):
        self.client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
        self.model_name = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
        self.vector_db = EmployeeVectorDB()
        self.predictor = AttritionPredictor()
        self.history = []

        # Direct system prompt - actionable output only
        self.system_prompt = """You are an HR retention expert. Output ONLY the following sections with bullet points:

**🎯 Priority Actions (This Week):**
• [Action 1]
• [Action 2]  
• [Action 3]

**💬 Manager Talking Points:**
• [Point 1]
• [Point 2]

**📊 30-Day Measurable Goal:**
• [Specific measurable outcome]

**💡 Quick Win:**
• [One immediate action that shows care]

DO NOT include any thinking, reasoning, or explanations. Start directly with the sections above."""

    def _extract_employee_id(self, msg: str):
        import re
        m = re.search(r"\b(\d{1,4})\b", msg)
        return int(m.group(1)) if m else None

    def _clean_llm_response(self, response: str) -> str:
        """Clean LLM response - remove <think>...</think> tags and any reasoning content."""
        if not response:
            return ""
        
        import re
        
        # 1. Remove <think>...</think> blocks (including multiline)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # 2. Remove other common thinking tag patterns
        response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'<reasoning>.*?</reasoning>', '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # 3. Strip and clean up extra whitespace
        response = response.strip()
        
        # 4. Remove lines that start with common thinking patterns (if any remain)
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            line_lower = line.lower().strip()
            # Skip lines that are clearly thinking/reasoning
            if line_lower.startswith(('okay,', 'let me', 'first,', 'so,', 'wait,', 'hmm', 'i need to', 'i should', 'the user', 'looking at')):
                continue
            cleaned_lines.append(line)
        
        response = '\n'.join(cleaned_lines).strip()
        
        # 5. Clean up multiple blank lines
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
        
        return response

    def _build_context_prompt(self, employee: dict, prediction: dict, context: dict) -> str:
        """Build context prompt for LLM."""
        
        dept_stats = context.get("department_stats", {})
        signals = ', '.join(prediction.get('risk_factors', ['No major risks']))
        
        prompt = f"""Analyze and provide retention recommendations for:

Employee #{employee.get('Employee_ID')} | {employee.get('Department')} | {employee.get('Job_Role')}
Risk Level: {prediction.get('risk_percentage')} ({prediction.get('risk_level')})
Key Signals: {signals}

Current Scores:
- Job Satisfaction: {employee.get('Job_Satisfaction')}/5
- Work-Life Balance: {employee.get('Work_Life_Balance')}/4
- Manager Relationship: {employee.get('Relationship_with_Manager')}/5
- Years Since Promotion: {employee.get('Years_Since_Last_Promotion')}
- Overtime: {employee.get('Overtime')}

Department Context: {dept_stats.get('attrition_rate', 0):.1f}% attrition rate

Provide specific, actionable recommendations in the required format."""
        return prompt

    # ===========================
    #  MAIN MESSAGE HANDLER
    # ===========================
    def process_message(self, user_msg: str):
        emp_id = self._extract_employee_id(user_msg)

        prediction = None
        employee = None
        structured_output = None

        if emp_id:
            # Get enhanced context from FAISS
            context = self.vector_db.get_enhanced_context(emp_id)
            employee = context.get("employee")
            
            if employee:
                prediction = self.predictor.predict(employee)
                
                # Generate structured actionable insights
                structured_output = format_retention_output(prediction, employee)
                
                # Build context prompt for LLM
                context_prompt = self._build_context_prompt(employee, prediction, context)
                
                # Get LLM response
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context_prompt}
                ]
                
                try:
                    out = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=1024,  # Increased for complete response
                        temperature=0.4
                        # Removed thinking config - let model think but we filter it out
                    )
                    raw_response = out.choices[0].message.content
                    llm_response = self._clean_llm_response(raw_response)
                except Exception as e:
                    llm_response = None
                
                # Return structured output with cleaned LLM enhancement
                return prediction, employee, structured_output, llm_response

        # Fallback for non-employee queries
        messages = [
            {"role": "system", "content": "You are an HR assistant. Provide helpful, concise answers about employee retention and HR topics."},
            {"role": "user", "content": user_msg}
        ]

        try:
            out = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=400,
                temperature=0.4
                # Removed thinking config - let model think but we filter it out
            )
            response = out.choices[0].message.content
            cleaned = self._clean_llm_response(response)
            return prediction, employee, None, cleaned or response

        except Exception as e:
            return prediction, employee, None, f"Error: {e}"

    def get_email_draft(self, employee: dict, template_type: str = "appreciation") -> str:
        """Get personalized email draft for an employee."""
        return generate_email_draft(template_type, employee)

    def clear_history(self):
        self.history = []
