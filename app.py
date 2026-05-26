import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="EduPro - Course Demand & Revenue Forecasting", layout="wide", page_icon="🎓")

st.markdown("""
<style>
    .main-header {font-size:2.2rem; font-weight:700; color:#1f4e79; text-align:center; padding:10px 0;}
    .metric-card {background:#f0f4ff; border-radius:10px; padding:15px; text-align:center; border-left:4px solid #1f4e79;}
    .section-header {font-size:1.4rem; font-weight:600; color:#1f4e79; border-bottom:2px solid #1f4e79; padding-bottom:5px; margin:20px 0 10px 0;}
</style>
""", unsafe_allow_html=True)

# ── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    users        = pd.read_csv("Users.csv")
    teachers     = pd.read_csv("Teachers.csv")
    courses      = pd.read_csv("Courses.csv")
    transactions = pd.read_csv("Transactions.csv")

    enroll       = transactions.groupby("CourseID").size().reset_index(name="EnrollmentCount")
    revenue      = transactions.groupby("CourseID")["Amount"].sum().reset_index(name="CourseRevenue")
    cat_revenue  = (transactions.merge(courses[["CourseID","CourseCategory"]], on="CourseID")
                                .groupby("CourseCategory")["Amount"].sum()
                                .reset_index(name="CategoryRevenue"))
    course_teacher = transactions.groupby("CourseID")["TeacherID"].first().reset_index()

    df = (courses
          .merge(enroll,        on="CourseID")
          .merge(revenue,       on="CourseID")
          .merge(course_teacher,on="CourseID")
          .merge(teachers[["TeacherID","YearsOfExperience","TeacherRating","Expertise"]], on="TeacherID"))
    return df, cat_revenue, users, transactions, courses

df, cat_revenue, users, transactions, courses = load_data()

# ── Feature Engineering ────────────────────────────────────────────────────────
def feature_engineering(df):
    d = df.copy()
    d["PriceBand"]       = pd.cut(d["CoursePrice"],   bins=[-1,0,300,600,10000], labels=["Free","Low","Medium","High"])
    d["DurationBucket"]  = pd.cut(d["CourseDuration"],bins=[0,15,30,60,1000],    labels=["Short","Medium","Long","XLong"])
    d["RatingTier"]      = pd.cut(d["CourseRating"],  bins=[0,2,3.5,5],          labels=["Low","Medium","High"])
    d["ExpBucket"]       = pd.cut(d["YearsOfExperience"], bins=[0,3,7,15,50],    labels=["Junior","Mid","Senior","Expert"])
    d["ExpertiseMatch"]  = (d["Expertise"] == d["CourseCategory"]).astype(int)
    return d

df = feature_engineering(df)

# ── ML Preparation ─────────────────────────────────────────────────────────────
@st.cache_data
def prepare_and_train(_df):
    d = _df.copy()
    le_cols = ["CourseCategory","CourseType","CourseLevel","PriceBand","DurationBucket","RatingTier","ExpBucket"]
    encoders = {}
    for col in le_cols:
        le = LabelEncoder()
        d[col+"_enc"] = le.fit_transform(d[col].astype(str))
        encoders[col] = le

    features = ["CoursePrice","CourseDuration","CourseRating","YearsOfExperience","TeacherRating",
                "ExpertiseMatch"] + [c+"_enc" for c in le_cols]

    results = {}
    for target in ["EnrollmentCount","CourseRevenue"]:
        X = d[features]
        y = d[target]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Linear Regression":      LinearRegression(),
            "Ridge":                  Ridge(alpha=1.0),
            "Lasso":                  Lasso(alpha=0.1),
            "Random Forest":          RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting":      GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        scores = {}
        for name, m in models.items():
            m.fit(X_tr, y_tr)
            pred = m.predict(X_te)
            scores[name] = {
                "MAE":  round(mean_absolute_error(y_te, pred), 2),
                "RMSE": round(np.sqrt(mean_squared_error(y_te, pred)), 2),
                "R2":   round(r2_score(y_te, pred), 4),
                "model": m,
                "X_test": X_te,
                "y_test": y_te,
            }
        results[target] = scores

    # Feature importance from best model (Random Forest on Enrollment)
    best_rf = results["EnrollmentCount"]["Random Forest"]["model"]
    importance_df = pd.DataFrame({"Feature": features, "Importance": best_rf.feature_importances_}).sort_values("Importance", ascending=False)

    return results, features, encoders, importance_df

results, features, encoders, importance_df = prepare_and_train(df)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/graduation-cap.png", width=80)
st.sidebar.title("EduPro Dashboard")
page = st.sidebar.radio("📌 Navigation", ["🏠 Overview", "📊 EDA", "🤖 ML Models", "🔍 Feature Importance", "🎯 Predict"])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<div class="main-header">🎓 EduPro — Course Demand & Revenue Forecasting</div>', unsafe_allow_html=True)
    st.markdown("### Platform Summary")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("👤 Total Users",       f"{len(users):,}")
    c2.metric("📚 Total Courses",     f"{len(courses):,}")
    c3.metric("💳 Transactions",      f"{len(transactions):,}")
    c4.metric("💰 Total Revenue",     f"₹{transactions['Amount'].sum():,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Revenue by Category</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7,4))
        cat_sorted = cat_revenue.sort_values("CategoryRevenue", ascending=True)
        bars = ax.barh(cat_sorted["CourseCategory"], cat_sorted["CategoryRevenue"],
                       color=plt.cm.Blues(np.linspace(0.4,0.9,len(cat_sorted))))
        ax.set_xlabel("Revenue (₹)")
        ax.set_title("Category-wise Revenue")
        for bar, val in zip(bars, cat_sorted["CategoryRevenue"]):
            ax.text(bar.get_width()+500, bar.get_y()+bar.get_height()/2,
                    f"₹{val:,.0f}", va='center', fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown('<div class="section-header">Course Type Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,4))
        ct = courses["CourseType"].value_counts()
        ax.pie(ct, labels=ct.index, autopct="%1.1f%%", colors=["#1f4e79","#2e86c1"],
               startangle=90, wedgeprops=dict(edgecolor='white'))
        ax.set_title("Free vs Paid Courses")
        st.pyplot(fig)

    st.markdown('<div class="section-header">Top 10 Courses by Enrollment</div>', unsafe_allow_html=True)
    top10 = df.nlargest(10, "EnrollmentCount")[["CourseName","CourseCategory","EnrollmentCount","CourseRevenue","CoursePrice"]]
    st.dataframe(top10.reset_index(drop=True), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown('<div class="main-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔗 Correlations", "📋 Category Deep Dive"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.hist(df["EnrollmentCount"], bins=20, color="#1f4e79", edgecolor="white")
            ax.set_title("Enrollment Count Distribution")
            ax.set_xlabel("Enrollments"); ax.set_ylabel("Frequency")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(6,4))
            paid = df[df["CoursePrice"]>0]["CourseRevenue"]
            ax.hist(paid, bins=20, color="#2e86c1", edgecolor="white")
            ax.set_title("Revenue Distribution (Paid Courses)")
            ax.set_xlabel("Revenue (₹)"); ax.set_ylabel("Frequency")
            st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.scatter(df["CoursePrice"], df["EnrollmentCount"], alpha=0.6, color="#1f4e79")
            ax.set_title("Price vs Enrollment")
            ax.set_xlabel("Course Price (₹)"); ax.set_ylabel("Enrollments")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.scatter(df["CourseRating"], df["EnrollmentCount"], alpha=0.6, color="#e74c3c")
            ax.set_title("Rating vs Enrollment")
            ax.set_xlabel("Course Rating"); ax.set_ylabel("Enrollments")
            st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(8,6))
        num_cols = ["CoursePrice","CourseDuration","CourseRating","YearsOfExperience","TeacherRating","EnrollmentCount","CourseRevenue"]
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax, linewidths=0.5)
        ax.set_title("Feature Correlation Heatmap")
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7,4))
            cat_enroll = df.groupby("CourseCategory")["EnrollmentCount"].mean().sort_values(ascending=False)
            ax.bar(cat_enroll.index, cat_enroll.values, color="#1f4e79")
            plt.xticks(rotation=45, ha='right', fontsize=8)
            ax.set_title("Avg Enrollment by Category")
            ax.set_ylabel("Avg Enrollments")
            plt.tight_layout()
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(7,4))
            level_rev = df.groupby("CourseLevel")["CourseRevenue"].mean().sort_values(ascending=False)
            ax.bar(level_rev.index, level_rev.values, color="#2e86c1")
            ax.set_title("Avg Revenue by Course Level")
            ax.set_ylabel("Avg Revenue (₹)")
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown('<div class="section-header">Price Band vs Enrollment</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8,4))
        pb = df.groupby("PriceBand")["EnrollmentCount"].mean()
        ax.bar(pb.index.astype(str), pb.values, color=["#27ae60","#f39c12","#e67e22","#e74c3c"])
        ax.set_title("Avg Enrollment by Price Band")
        ax.set_ylabel("Avg Enrollments")
        st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ML MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Models":
    st.markdown('<div class="main-header">🤖 Machine Learning Models</div>', unsafe_allow_html=True)

    target_choice = st.selectbox("Select Target Variable", ["EnrollmentCount", "CourseRevenue"])
    scores = results[target_choice]

    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
    rows = []
    for model_name, s in scores.items():
        rows.append({"Model": model_name, "MAE": s["MAE"], "RMSE": s["RMSE"], "R² Score": s["R2"]})
    score_df = pd.DataFrame(rows).sort_values("R² Score", ascending=False)
    st.dataframe(score_df.reset_index(drop=True), use_container_width=True)

    best_model_name = score_df.iloc[0]["Model"]
    best_r2 = score_df.iloc[0]["R² Score"]
    st.success(f"✅ Best Model: **{best_model_name}** — R² Score: **{best_r2}**")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(score_df["Model"], score_df["R² Score"], color=plt.cm.Blues(np.linspace(0.4,0.9,5)))
        plt.xticks(rotation=30, ha='right', fontsize=8)
        ax.set_title(f"R² Score Comparison — {target_choice}")
        ax.set_ylabel("R² Score")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        st.pyplot(fig)
    with col2:
        best = scores[best_model_name]
        y_pred = best["model"].predict(best["X_test"])
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(best["y_test"], y_pred, alpha=0.6, color="#1f4e79")
        mn = min(best["y_test"].min(), y_pred.min())
        mx = max(best["y_test"].max(), y_pred.max())
        ax.plot([mn,mx],[mn,mx], 'r--', lw=2)
        ax.set_title(f"Actual vs Predicted — {best_model_name}")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        plt.tight_layout()
        st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Feature Importance":
    st.markdown('<div class="main-header">🔍 Feature Importance Analysis</div>', unsafe_allow_html=True)
    st.markdown("Based on **Random Forest** model for Enrollment Count prediction.")

    col1, col2 = st.columns([3,2])
    with col1:
        fig, ax = plt.subplots(figsize=(8,6))
        top_features = importance_df.head(10)
        bars = ax.barh(top_features["Feature"][::-1], top_features["Importance"][::-1],
                       color=plt.cm.Blues(np.linspace(0.4,0.9,10)))
        ax.set_title("Top 10 Feature Importances")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        st.pyplot(fig)
    with col2:
        st.markdown("### 💡 Business Insights")
        st.info("📌 **CourseRating** is the biggest driver of enrollment demand.")
        st.info("📌 **CoursePrice** significantly affects revenue predictions.")
        st.info("📌 **TeacherRating** influences course popularity.")
        st.info("📌 **CourseDuration** affects learner decisions.")
        st.info("📌 **CourseLevel** determines target audience size.")

    st.markdown("### Full Feature Importance Table")
    st.dataframe(importance_df.reset_index(drop=True), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Predict":
    st.markdown('<div class="main-header">🎯 Predict Enrollment & Revenue</div>', unsafe_allow_html=True)
    st.markdown("Enter course details to predict demand and revenue.")

    col1, col2, col3 = st.columns(3)
    with col1:
        price    = st.slider("💰 Course Price (₹)", 0, 1000, 300)
        duration = st.slider("⏱️ Duration (hours)",  1,  100,  20)
        c_rating = st.slider("⭐ Course Rating",    1.0, 5.0, 4.0, 0.1)
    with col2:
        yoe      = st.slider("👨‍🏫 Teacher Experience (yrs)", 1, 20, 5)
        t_rating = st.slider("⭐ Teacher Rating",  1.0, 5.0, 3.5, 0.1)
        exp_match = st.selectbox("🎯 Expertise Match", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
    with col3:
        category = st.selectbox("📂 Category",   sorted(df["CourseCategory"].unique()))
        ctype    = st.selectbox("🆓 Type",        ["Paid","Free"])
        level    = st.selectbox("📊 Level",       ["Beginner","Intermediate","Advanced"])

    price_band     = "Free" if price==0 else ("Low" if price<300 else ("Medium" if price<600 else "High"))
    dur_bucket     = "Short" if duration<15 else ("Medium" if duration<30 else ("Long" if duration<60 else "XLong"))
    rating_tier    = "Low" if c_rating<2 else ("Medium" if c_rating<3.5 else "High")
    exp_bucket     = "Junior" if yoe<=3 else ("Mid" if yoe<=7 else ("Senior" if yoe<=15 else "Expert"))

    def encode_val(col, val):
        try:    return int(encoders[col].transform([val])[0])
        except: return 0

    input_data = pd.DataFrame([{
        "CoursePrice": price, "CourseDuration": duration, "CourseRating": c_rating,
        "YearsOfExperience": yoe, "TeacherRating": t_rating, "ExpertiseMatch": exp_match,
        "CourseCategory_enc": encode_val("CourseCategory", category),
        "CourseType_enc":     encode_val("CourseType",     ctype),
        "CourseLevel_enc":    encode_val("CourseLevel",    level),
        "PriceBand_enc":      encode_val("PriceBand",      price_band),
        "DurationBucket_enc": encode_val("DurationBucket", dur_bucket),
        "RatingTier_enc":     encode_val("RatingTier",     rating_tier),
        "ExpBucket_enc":      encode_val("ExpBucket",      exp_bucket),
    }])

    if st.button("🚀 Predict Now", use_container_width=True):
        best_enroll  = max(results["EnrollmentCount"],  key=lambda k: results["EnrollmentCount"][k]["R2"])
        best_revenue = max(results["CourseRevenue"],    key=lambda k: results["CourseRevenue"][k]["R2"])

        pred_enroll  = results["EnrollmentCount"][best_enroll]["model"].predict(input_data)[0]
        pred_revenue = results["CourseRevenue"][best_revenue]["model"].predict(input_data)[0]

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("👥 Predicted Enrollments", f"{max(0,int(pred_enroll)):,}")
        c2.metric("💰 Predicted Revenue",     f"₹{max(0,pred_revenue):,.0f}")
        c3.metric("📊 Category",              category)

        st.success(f"✅ Used **{best_enroll}** for enrollment and **{best_revenue}** for revenue prediction.")
