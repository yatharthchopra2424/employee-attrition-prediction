# ================================
#  NVIDIA RAG Chatbot (Updated)
#  Structured concise retention output
# ================================

import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from encoding_map import ENCODING_MAP
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
# 🔥 New: Formatter to enforce structured output
# ==========================================================

def format_retention_output(pred: dict) -> str:
    """Return forced concise structured output, ignoring LLM rambling."""

    risk = pred.get("risk_percentage", "N/A")
    signals = pred.get("risk_factors", [])[:3]
    signals_txt = ", ".join(s.replace("⚠️ ", "") for s in signals) or "None"

    # Core recommended actions based on signals
    actions = []
    if "work-life" in signals_txt.lower():
        actions.append("flexible scheduling")
    if "overtime" in signals_txt.lower():
        actions.append("overtime reduction plan")
    if "compensation" in signals_txt.lower():
        actions.append("compensation review")
    if not actions:
        actions = ["manager check-ins", "growth pathway"]

    actions_txt = ", ".join(actions[:3])

    impact = "reduced attrition risk via targeted interventions"

    return (
        f"Predicted risk: {risk}\n"
        f"Key signals: {signals_txt}\n"
        f"Recommended actions: {actions_txt}\n"
        f"Expected impact: {impact}"
    )


# ==========================================================
# FAISS Vector DB (UNTOUCHED, just kept clean)
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
        return (
            f"Employee {row.get('Employee_ID')} Age {row.get('Age')} "
            f"{row.get('Gender')} {row.get('Marital_Status')} "
            f"Department {row.get('Department')} Role {row.get('Job_Role')} "
            f"Income {row.get('Monthly_Income')} Years {row.get('Years_at_Company')} "
            f"Satisfaction {row.get('Job_Satisfaction')} WorkLife {row.get('Work_Life_Balance')} "
            f"Overtime {row.get('Overtime')} Manager {row.get('Relationship_with_Manager')}"
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


# ==========================================================
# ML Predictor (unchanged except clean)
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

            # risk factors
            f = []
            if row.get("Work_Life_Balance", 5) <= 2: f.append("poor work-life balance")
            if row.get("Overtime") == "Yes": f.append("frequent overtime")
            if row.get("Monthly_Income", 10000) < 6000: f.append("below-average compensation")
            if row.get("Relationship_with_Manager", 5) <= 2: f.append("weak manager relationship")

            return {
                "success": True,
                "employee_id": row.get("Employee_ID"),
                "risk_probability": float(p),
                "risk_percentage": f"{p*100:.1f}%",
                "risk_level": level,
                "color": color,
                "risk_factors": f[:3]
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
# 🤖 NVIDIARAGChatbot — Updated with FORCED concise output
# ==========================================================

class NVIDIARAGChatbot:

    def __init__(self):
        self.client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
        self.model_name = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
        self.vector_db = EmployeeVectorDB()
        self.predictor = AttritionPredictor()
        self.history = []

        # UPDATED SYSTEM PROMPT
        self.system_prompt = """
You are a retention analyst. 
You must ALWAYS output in this exact 4-line format:

Predicted risk: <risk%>
Key signals: <comma-separated signals>
Recommended actions: <comma-separated actions>
Expected impact: <1 concise impact line>

No extra sentences. No greetings. No explanations.
"""

    def _extract_employee_id(self, msg: str):
        import re
        m = re.search(r"\b(\d{1,4})\b", msg)
        return int(m.group(1)) if m else None

    # ===========================
    #  MAIN MESSAGE HANDLER
    # ===========================
    def process_message(self, user_msg: str):
        emp_id = self._extract_employee_id(user_msg)

        prediction = None
        employee = None

        if emp_id:
            employee = self.vector_db.get_employee_by_id(emp_id)
            if employee:
                prediction = self.predictor.predict(employee)

                # ❗ New: skip LLM entirely → ALWAYS return structured summary
                response = format_retention_output(prediction)
                return prediction, employee, response

        # fallback LLM response (rare case)
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_msg}]

        try:
            out = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=150,
                temperature=0.2,
                extra_body={"thinking": {"type": "disabled"}}
            )
            response = out.choices[0].message.content
            return prediction, employee, response

        except Exception as e:
            return prediction, employee, f"Error: {e}"

    def clear_history(self):
        self.history = []
