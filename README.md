# 🎓 EduPro - Predictive Modeling for Course Demand & Revenue Forecasting

## Project Overview
Internship project for Unified Mentor Pvt. Ltd., Gurugram.
Platform: EduPro Online Learning | College: NC Auto College, Jajpur

## Problem Statement
EduPro lacks predictive models for:
- Course enrollment demand
- Revenue forecasting at course & category level
- Data-driven course launch & pricing decisions

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (ML Models)
- Matplotlib, Seaborn (Visualizations)
- Streamlit (Web App)

## Dataset
- `data/Users.csv` — 3000 users
- `data/Teachers.csv` — 60 teachers
- `data/Courses.csv` — 60 courses
- `data/Transactions.csv` — 10,000 transactions

## ML Models Used
- Linear Regression (Baseline)
- Ridge Regression
- Lasso Regression
- Random Forest Regressor ✅ (Best)
- Gradient Boosting Regressor

## Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

## How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Streamlit app
```bash
streamlit run app.py
```

### Step 3: Open browser
Go to: http://localhost:8501

## App Pages
1. 🏠 **Overview** — Platform KPIs, top courses, revenue by category
2. 📊 **EDA** — Distributions, correlations, category analysis
3. 🤖 **ML Models** — Compare 5 models, actual vs predicted plots
4. 🔍 **Feature Importance** — Key demand drivers analysis
5. 🎯 **Predict** — Real-time enrollment & revenue prediction

## Deliverables
- ✅ Streamlit Dashboard (live analytics)
- ✅ ML Models (Enrollment + Revenue prediction)
- ✅ Feature Importance Analysis
- ✅ EDA Visualizations
