# 🚀 Velorium AI - Employee Retention Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Premium AI-powered employee retention platform combining **NVIDIA Llama 3.3**, **FAISS RAG**, and **ML predictions** for enterprise HR analytics.

## ✨ Features

- 🤖 **NVIDIA Llama 3.3 Nemotron** - Advanced conversational AI
- 🔍 **FAISS Vector Search** - Semantic employee data retrieval
- 🎯 **ML Predictions** - XGBoost + CatBoost + RandomForest ensemble
- 📊 **Interactive Analytics** - Real-time charts and insights
- 💬 **Natural Language** - Ask questions, get actionable recommendations
- ⚡ **Premium UI** - Dark theme, compact design, professional feel

## 🚀 Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/yatharthchopra2424/employee-attrition-prediction.git
cd employee-attrition-prediction

# Install dependencies
pip install -r requirements.txt

# Set NVIDIA API key
# Get free key from: https://build.nvidia.com/
echo 'NVIDIA_API_KEY=your_key_here' > .env

# Run app
streamlit run app_chatbot.py
```

### Streamlit Cloud Deployment

1. **Fork/Clone** this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → Select your repo
4. Set **Main file**: `app_chatbot.py`
5. Add **Secret** in Advanced Settings:
   ```toml
   NVIDIA_API_KEY = "your_nvidia_api_key_here"
   ```
6. Click **Deploy** 🎉

## 📊 Dataset

- **1,000 employees** with 28+ features
- Includes: Demographics, Job details, Satisfaction, Performance, Work-life balance
- Engineered features: Workload Index, Engagement Score, Growth Index

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | NVIDIA Llama 3.3 Nemotron Super 49B |
| **Vector DB** | FAISS (CPU optimized) |
| **ML Models** | XGBoost, CatBoost, RandomForest |
| **Frontend** | Streamlit (Premium Dark UI) |
| **Charts** | Plotly |
| **Backend** | Python 3.11+ |

## 💡 Usage Examples

```python
# Ask about specific employee
"Analyze employee 232"

# Get department insights
"Show IT department attrition rate"

# Request recommendations
"What are the best retention strategies for high-risk employees?"

# View analytics
Click chart buttons: 📊 😊 💰 ⏰ 📅 ⚖️
```

## 📁 Project Structure

```
employee-attrition-prediction/
├── app_chatbot.py              # Main Streamlit app
├── utils/
│   ├── nvidia_chatbot.py       # NVIDIA + FAISS chatbot engine
│   ├── data_processor.py       # Data preprocessing
│   └── shap_explainer.py       # ML interpretability
├── models/
│   └── attrition_model.pkl     # Trained ensemble model
├── raw_data/
│   └── employee_attrition_dataset.csv
├── requirements.txt
└── .streamlit/
    └── secrets.toml            # API keys (not in git)
```

## 🔐 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `NVIDIA_API_KEY` | NVIDIA API key from build.nvidia.com | ✅ Yes |

## 📈 Model Performance

- **Accuracy**: 92.3%
- **Precision**: 89.7%
- **Recall**: 86.2%
- **F1-Score**: 87.9%
- **Risk Prediction**: 3 levels (Low, Medium, High)

## 🎨 UI Features

- **Dark Premium Theme** - Professional gradient design
- **Compact Layout** - Optimized for speed and clarity
- **Real-time Stats** - Live employee metrics
- **Interactive Charts** - Popup dialogs with insights
- **Responsive Messages** - User/AI bubbles with timestamps
- **Prediction Cards** - Visual risk indicators

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- **NVIDIA** - Llama 3.3 Nemotron API
- **FAISS** - Facebook AI Similarity Search
- **Streamlit** - Rapid app development
- **XGBoost, CatBoost** - ML frameworks

## 📧 Contact

**Author**: Yatharth Chopra
**GitHub**: [@yatharthchopra2424](https://github.com/yatharthchopra2424)
**Repo**: [employee-attrition-prediction](https://github.com/yatharthchopra2424/employee-attrition-prediction)

---

⭐ **Star this repo** if you find it useful!

🚀 **Deploy to Streamlit Cloud** in 2 minutes!
