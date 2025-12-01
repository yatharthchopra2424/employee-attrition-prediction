"""
NVIDIA RAG Chatbot - FAISS + ML Model Integration
Fast, concise responses with automatic predictions
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
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

# NVIDIA API KEY - HARDCODED
NVIDIA_API_KEY = "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"

# Model expects these 28 features in this exact order
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


class EmployeeVectorDB:
    """FAISS-based Vector Database for Employee Data"""
    
    def __init__(self, csv_path: str = "raw_data/employee_attrition_dataset.csv"):
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
                print(f"✓ Loaded {len(self.df)} employee records")
        except Exception as e:
            print(f"⚠ Error loading data: {e}")
    
    def _create_text_representation(self, row: pd.Series) -> str:
        return f"""Employee {row.get('Employee_ID', 'N/A')} Age {row.get('Age', 'N/A')} 
        {row.get('Gender', 'N/A')} {row.get('Marital_Status', 'N/A')} 
        Department {row.get('Department', 'N/A')} Role {row.get('Job_Role', 'N/A')} 
        Income {row.get('Monthly_Income', 'N/A')} Years {row.get('Years_at_Company', 'N/A')} 
        Satisfaction {row.get('Job_Satisfaction', 'N/A')} WorkLife {row.get('Work_Life_Balance', 'N/A')} 
        Overtime {row.get('Overtime', 'N/A')} Manager {row.get('Relationship_with_Manager', 'N/A')}"""
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        text_lower = text.lower()
        for i in range(len(text_lower) - 2):
            trigram = text_lower[i:i+3]
            hash_val = hash(trigram) % self.embedding_dim
            embedding[hash_val] += 1
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    
    def _build_index(self):
        if self.df is None or self.df.empty:
            return
        print("📊 Building FAISS vector index...")
        embeddings_list = []
        for _, row in self.df.iterrows():
            text = self._create_text_representation(row)
            embedding = self._simple_embedding(text)
            embeddings_list.append(embedding)
        self.embeddings = np.array(embeddings_list, dtype=np.float32)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(self.embeddings)
        print(f"✓ FAISS index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not FAISS_AVAILABLE or self.index is None:
            return self._keyword_search(query, top_k)
        query_embedding = self._simple_embedding(query).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.df):
                record = self.df.iloc[idx].to_dict()
                record['_score'] = float(1 / (1 + distances[0][i]))
                results.append(record)
        return results
    
    def _keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.df is None:
            return []
        query_lower = query.lower()
        scores = []
        for idx, row in self.df.iterrows():
            text = self._create_text_representation(row).lower()
            score = sum(1 for word in query_lower.split() if word in text)
            scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [self.df.iloc[idx].to_dict() for idx, _ in scores[:top_k]]
    
    def get_employee_by_id(self, emp_id: int) -> Optional[Dict]:
        if self.df is None:
            return None
        emp = self.df[self.df['Employee_ID'] == emp_id]
        return emp.iloc[0].to_dict() if not emp.empty else None
    
    def get_department_stats(self, department: str) -> Dict:
        if self.df is None:
            return {}
        dept_df = self.df[self.df['Department'] == department]
        if dept_df.empty:
            return {}
        return {
            'total': len(dept_df),
            'avg_income': round(dept_df['Monthly_Income'].mean(), 0),
            'avg_satisfaction': round(dept_df['Job_Satisfaction'].mean(), 2),
            'overtime_rate': round((dept_df['Overtime'] == 'Yes').mean() * 100, 1),
            'attrition_rate': round((dept_df['Attrition'] == 'Yes').mean() * 100, 1),
        }


class AttritionPredictor:
    """ML Model Predictor - Fixed to handle dictionary model format"""
    
    def __init__(self, model_path: str = "models/attrition_model.pkl"):
        self.model = None
        self.columns = MODEL_FEATURES  # Default to our known features
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Model is stored as dict with 'model' and 'columns' keys
                if isinstance(data, dict):
                    self.model = data.get('model')
                    self.columns = data.get('columns', MODEL_FEATURES)
                    print(f"✓ ML model loaded (dict format, {len(self.columns)} features)")
                else:
                    # Direct model object
                    self.model = data
                    print(f"✓ ML model loaded (direct format)")
            else:
                print(f"⚠ Model file not found: {model_path}")
        except Exception as e:
            print(f"⚠ Model error: {e}")
    
    def _prepare_features(self, data: Dict) -> List:
        """Prepare exactly 28 features in the correct order"""
        d = data.copy()
        
        # === COMPUTE ENGINEERED FEATURES ===
        
        # Distance_Bucket (from Distance_From_Home)
        dist = d.get('Distance_From_Home', 10)
        if dist <= 10:
            d['Distance_Bucket'] = 0  # Close
        elif dist <= 20:
            d['Distance_Bucket'] = 2  # Medium
        elif dist <= 30:
            d['Distance_Bucket'] = 1  # Far
        else:
            d['Distance_Bucket'] = 3  # Very Far
        
        # Age_Group (from Age)
        age = d.get('Age', 35)
        if age < 30:
            d['Age_Group'] = 0  # <30
        elif age < 40:
            d['Age_Group'] = 1  # 30-40
        elif age < 50:
            d['Age_Group'] = 2  # 40-50
        else:
            d['Age_Group'] = 3  # >50
        
        # Tenure_Bucket (from Years_at_Company)
        years = d.get('Years_at_Company', 5)
        if years < 3:
            d['Tenure_Bucket'] = 0  # Early
        elif years < 10:
            d['Tenure_Bucket'] = 2  # Mid
        else:
            d['Tenure_Bucket'] = 1  # Long
        
        # Income_Level (from Monthly_Income)
        income = d.get('Monthly_Income', 10000)
        if income < 8000:
            d['Income_Level'] = 0  # Q1
        elif income < 12000:
            d['Income_Level'] = 1  # Q2
        elif income < 16000:
            d['Income_Level'] = 2  # Q3
        else:
            d['Income_Level'] = 3  # Q4
        
        # Absenteeism_Level (from Absenteeism)
        absent = d.get('Absenteeism', 5)
        if absent < 5:
            d['Absenteeism_Level'] = 1  # Low
        elif absent < 15:
            d['Absenteeism_Level'] = 2  # Medium
        else:
            d['Absenteeism_Level'] = 0  # High
        
        # Overtime_Flag (from Overtime)
        overtime = d.get('Overtime', 'No')
        d['Overtime_Flag'] = 1 if overtime in ['Yes', 1, True, 'yes'] else 0
        
        # Workload_Index
        avg_hours = d.get('Average_Hours_Worked_Per_Week', 45)
        project_count = d.get('Project_Count', 3)
        d['Workload_Index'] = (avg_hours * project_count) / 100
        
        # Engagement_Score
        job_sat = d.get('Job_Satisfaction', 3)
        job_inv = d.get('Job_Involvement', 3)
        work_env = d.get('Work_Environment_Satisfaction', 3)
        d['Engagement_Score'] = (job_sat + job_inv + work_env) / 15 * 5
        
        # Growth_Index
        perf_rating = d.get('Performance_Rating', 3)
        training_hrs = d.get('Training_Hours_Last_Year', 20)
        years_since_promo = max(d.get('Years_Since_Last_Promotion', 1), 1)
        d['Growth_Index'] = (perf_rating + training_hrs / 10) / years_since_promo
        
        # Training_Effectiveness
        years_at_company = max(d.get('Years_at_Company', 1), 1)
        d['Training_Effectiveness'] = training_hrs / years_at_company
        
        # Work_Efficiency
        d['Work_Efficiency'] = (perf_rating / max(avg_hours, 1)) * 10
        
        # === ENCODE CATEGORICAL FEATURES ===
        # Gender
        gender = d.get('Gender', 'Male')
        d['Gender'] = 1 if gender in ['Male', 1] else 0
        
        # Marital_Status
        marital = d.get('Marital_Status', 'Single')
        marital_map = {'Single': 2, 'Married': 1, 'Divorced': 0}
        d['Marital_Status'] = marital_map.get(marital, 2) if isinstance(marital, str) else marital
        
        # Department
        dept = d.get('Department', 'IT')
        dept_map = {'IT': 1, 'Sales': 4, 'HR': 0, 'Marketing': 2, 'R&D': 3, 'Finance': 0, 'Operations': 2}
        d['Department'] = dept_map.get(dept, 1) if isinstance(dept, str) else dept
        
        # Job_Role
        role = d.get('Job_Role', 'Developer')
        role_map = {
            'Developer': 1, 'Manager': 4, 'Analyst': 0, 'Executive': 2, 
            'Sales Rep': 7, 'Engineer': 2, 'Consultant': 1, 'Director': 1,
            'Researcher': 6, 'Technician': 8, 'Specialist': 8
        }
        d['Job_Role'] = role_map.get(role, 1) if isinstance(role, str) else role
        
        # === BUILD FEATURE VECTOR IN CORRECT ORDER ===
        features = []
        for feat in MODEL_FEATURES:
            val = d.get(feat, 0)
            # Ensure numeric
            if isinstance(val, str):
                val = 0
            features.append(float(val))
        
        return features
    
    def predict(self, employee_data: Dict) -> Dict:
        if self.model is None:
            return {
                'success': False, 
                'error': 'Model not loaded', 
                'risk_level': 'UNKNOWN', 
                'risk_probability': 0,
                'risk_percentage': 'N/A',
                'color': '⚪',
                'risk_factors': []
            }
        
        try:
            features = self._prepare_features(employee_data)
            features_array = np.array([features])
            
            pred = self.model.predict(features_array)[0]
            
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features_array)[0]
                risk_proba = proba[1] if len(proba) > 1 else proba[0]
            else:
                risk_proba = float(pred)
            
            if risk_proba >= 0.7:
                level, color = "HIGH", "🔴"
            elif risk_proba >= 0.4:
                level, color = "MEDIUM", "🟡"
            else:
                level, color = "LOW", "🟢"
            
            factors = self._get_risk_factors(employee_data)
            
            return {
                'success': True,
                'employee_id': employee_data.get('Employee_ID', 'N/A'),
                'risk_probability': float(risk_proba),
                'risk_percentage': f"{risk_proba * 100:.1f}%",
                'risk_level': level,
                'color': color,
                'risk_factors': factors
            }
        except Exception as e:
            import traceback
            print(f"Prediction error: {e}")
            traceback.print_exc()
            return {
                'success': False, 
                'error': str(e), 
                'risk_level': 'ERROR', 
                'risk_probability': 0,
                'risk_percentage': 'N/A',
                'color': '⚪',
                'risk_factors': []
            }
    
    def _get_risk_factors(self, data: Dict) -> List[str]:
        factors = []
        if data.get('Work_Life_Balance', 5) <= 2:
            factors.append("⚠️ Poor work-life balance")
        if data.get('Job_Satisfaction', 5) <= 2:
            factors.append("⚠️ Low job satisfaction")
        if data.get('Overtime') == 'Yes':
            factors.append("⚠️ Frequent overtime")
        if data.get('Years_Since_Last_Promotion', 0) > 3:
            factors.append("⚠️ No promotion in 3+ years")
        if data.get('Work_Environment_Satisfaction', 5) <= 2:
            factors.append("⚠️ Poor work environment")
        if data.get('Relationship_with_Manager', 5) <= 2:
            factors.append("⚠️ Weak manager relationship")
        if data.get('Monthly_Income', 10000) < 6000:
            factors.append("⚠️ Below average compensation")
        return factors[:5]


class NVIDIARAGChatbot:
    """NVIDIA RAG Chatbot with FAISS + ML Integration"""
    
    def __init__(self):
        self.client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
        self.model_name = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
        self.vector_db = EmployeeVectorDB()
        self.predictor = AttritionPredictor()
        self.history = []
        
        # CONCISE system prompt - no verbose responses, no thinking
        self.system_prompt = """You are a concise HR retention assistant. 

STRICT RULES:
1. For greetings (hi, hello): Reply in 1 sentence only.
2. For employee queries: Give ONLY 2-3 bullet point recommendations.
3. NEVER show your thinking process or reasoning.
4. NEVER say "let me think", "okay", "first I need to", etc.
5. Go straight to the answer - no preamble.
6. Be direct and professional.
7. Maximum 3-4 sentences total.

Example good response:
• Implement flexible work arrangements (hybrid/remote options)
• Create a career development plan with promotion timeline
• Schedule regular recognition and feedback sessions"""
    
    def _is_greeting(self, msg: str) -> bool:
        """Check if message is a simple greeting"""
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'howdy', 'greetings']
        msg_clean = msg.lower().strip().rstrip('!.,?')
        return msg_clean in greetings or len(msg_clean) <= 3
    
    def _extract_employee_id(self, msg: str) -> Optional[int]:
        import re
        patterns = [r'employee\s*(?:id\s*)?(\d+)', r'emp\s*(\d+)', r'#(\d+)', r'\b(\d{1,4})\b']
        for p in patterns:
            m = re.search(p, msg.lower())
            if m:
                eid = int(m.group(1))
                if 1 <= eid <= 1000:
                    return eid
        return None
    
    def process_message(self, user_msg: str) -> Tuple[Optional[Dict], Optional[Dict], str]:
        """Process message: returns (prediction, employee_data, response)"""
        self.history.append({"role": "user", "content": user_msg})
        
        # Handle simple greetings quickly
        if self._is_greeting(user_msg):
            response = "Hello! I'm here to help with employee retention analysis. Ask about any employee (e.g., 'analyze employee 232') or general HR questions."
            self.history.append({"role": "assistant", "content": response})
            return None, None, response
        
        prediction = None
        employee = None
        context = ""
        
        emp_id = self._extract_employee_id(user_msg)
        
        if emp_id:
            employee = self.vector_db.get_employee_by_id(emp_id)
            if employee:
                prediction = self.predictor.predict(employee)
                context = f"""Employee {emp_id}: {employee.get('Department')} {employee.get('Job_Role')}, ${employee.get('Monthly_Income')}/mo, {employee.get('Years_at_Company')}yrs
ML Prediction: {prediction.get('risk_level')} risk ({prediction.get('risk_percentage')})
Top factors: {', '.join(prediction.get('risk_factors', ['None identified'])[:3])}
Give 2-3 specific retention recommendations."""
        else:
            results = self.vector_db.search(user_msg, top_k=3)
            if results:
                context = "Relevant employees:\n" + "\n".join(
                    f"- #{e['Employee_ID']}: {e['Department']} {e['Job_Role']}" for e in results
                )
        
        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history[-4:])  # Keep history short
        if context:
            messages[-1] = {"role": "user", "content": user_msg + "\n\n" + context}
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                top_p=0.9,
                max_tokens=500,
                stream=False,
                extra_body={"thinking": {"type": "disabled"}}
            )
            response = completion.choices[0].message.content
            self.history.append({"role": "assistant", "content": response})
            return prediction, employee, response
        except Exception as e:
            return prediction, employee, f"Error: {e}"
    
    def stream_response(self, user_msg: str):
        """Stream response with prediction first"""
        
        # Handle greetings without streaming
        if self._is_greeting(user_msg):
            yield {"type": "text", "data": "Hello! I'm here to help with employee retention analysis. Ask about any employee (e.g., 'analyze employee 232') or general HR questions."}
            return
        
        emp_id = self._extract_employee_id(user_msg)
        prediction = None
        employee = None
        context = ""
        
        if emp_id:
            employee = self.vector_db.get_employee_by_id(emp_id)
            if employee:
                prediction = self.predictor.predict(employee)
                yield {"type": "prediction", "data": prediction, "employee": employee}
                context = f"""Employee {emp_id}: {employee.get('Department')} {employee.get('Job_Role')}
Risk: {prediction.get('risk_level')} ({prediction.get('risk_percentage')})
Factors: {', '.join(prediction.get('risk_factors', [])[:3])}
Give 2-3 specific recommendations."""
        else:
            results = self.vector_db.search(user_msg, top_k=3)
            if results:
                context = "Data:\n" + "\n".join(f"#{e['Employee_ID']}: {e['Department']}" for e in results)
        
        self.history.append({"role": "user", "content": user_msg})
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history[-4:])
        if context:
            messages[-1] = {"role": "user", "content": user_msg + "\n" + context}
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name, 
                messages=messages,
                temperature=0.3, 
                top_p=0.9, 
                max_tokens=500, 
                stream=True,
                extra_body={"thinking": {"type": "disabled"}}
            )
            full = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    c = chunk.choices[0].delta.content
                    full += c
                    yield {"type": "text", "data": c}
            self.history.append({"role": "assistant", "content": full})
        except Exception as e:
            yield {"type": "error", "data": str(e)}
    
    def clear_history(self):
        self.history = []
