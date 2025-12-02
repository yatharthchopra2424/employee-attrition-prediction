# Velorium — Employee Retention Copilot

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Powered-orange?style=for-the-badge)](https://xgboost.readthedocs.io/)


**Status:** Prototype / MVP

---

## Purpose
This repository implements a per-employee attrition risk prediction and a small Streamlit-based MVP for exploring predictions, SHAP explanations and generative recommendations. It is intended as a case-study / manager-facing prototype.

---

## Quick Start (Windows PowerShell)

- **Prerequisites:** Python 3.9+ and `pip` installed.
- **Install dependencies:**
```powershell
pip install -r requirements.txt
```
- **Run the Streamlit MVP:**
```powershell
streamlit run app_mvp.py
```
- Open the app at `http://localhost:8501`.

Notes:
- The generative AI features require API keys (optional). See "GenAI configuration" below.
- If `models/attrition_model.pkl` is not present, train the model using the included Jupyter notebook (see "Train the model").

---

## Repository Layout
Top-level files and folders you will interact with:

- `app_mvp.py` — Streamlit MVP application (main entrypoint for demo)
- `app_chatbot.py` — Alternative chat interface / assistant (experimental)
- `app.py` — Original / archive app variant
- `requirements.txt` — Python dependency manifest
- `jyputer_notebook/` — Training & exploration notebook (`Employee_Attrition_Risk_Prediciton_V5_ipynb.ipynb`)
- `models/` — Saved model artifacts (e.g., `attrition_model.pkl`)
- `raw_data/` — Original dataset (`employee_attrition_dataset.csv`)
- `utils/` — Helper modules:
  - `data_processor.py` — data loading & preprocessing
  - `encoding_map.py` — categorical encoding maps
  - `shap_explainer.py` — SHAP utilities and plotting helpers
  - `genai_engine.py` — LLM prompt wrappers (optional)

If you maintain this repo, keep `models/` and `raw_data/` out of version control or store them in a release/ARTIFACT store if large.

---

## GenAI configuration (optional)

To enable LLM-powered recommendations, add API keys as environment variables. Create a `.env` file (local only).

- Groq (recommended, free): `GROQ_API_KEY=...`
- OpenAI (optional): `OPENAI_API_KEY=...`

If you use a `.env` file, load it securely in your environment (do not commit your keys).

---

## Train the model

If `models/attrition_model.pkl` is missing or you want to retrain:

1. Open the notebook: `jyputer_notebook/Employee_Attrition_Risk_Prediciton_V5_ipynb.ipynb` and run the cells to preprocess, train and save the model.
2. The notebook writes the final model artifact to `models/` (check the notebook for exact filename).

Tip: run the notebook inside a virtual environment matching `requirements.txt`.

---

## Using the code

- Use `utils/data_processor.py` to load and preprocess inputs for prediction.
- Load trained model in your scripts with joblib / pickle (see notebook for example usage).
- Use `utils/shap_explainer.py` to compute and visualize SHAP explanations for individual predictions.

---

## Development & Contribution

- Fork the repository and create feature branches for PRs.
- Keep changes focused and include tests/snippets when adding model or API behavior.

Suggested workflow (PowerShell):
```powershell
git checkout -b feat/your-feature
# Make changes
git add .
git commit -m "feat: short description"
git push origin feat/your-feature
```

---

## Recommended next steps

- Add a minimal `requirements.txt` lock or `pyproject.toml` for reproducible installs.
- Add a small `CONTRIBUTING.md` with code style and PR guidance if you plan to accept contributions.

---

## Contact & License

For questions or issues, open an issue in the repository. This project is provided as a case-study; include a license file if you plan to publish or share publicly.

---

*Prepared for internal evaluation and demo.*
