import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, r2_score
from utils.preprocess import preprocess_churn, preprocess_tenure
import numpy as np

st.set_page_config(layout="wide")
st.title("📊 Customer Churn & Tenure Prediction Dashboard")

# Load models
churn_model = joblib.load("models/churn_logreg_pipeline.pkl")
tenure_model = joblib.load("models/tenure_pipeline.pkl")

# File uploader
st.markdown("### 📝 Data Requirements")
st.markdown("""
- File must be in **CSV format**.
- Must include columns like: `gender`, `Contract`, `PaymentMethod`, `InternetService`, `SeniorCitizen`, `tenure`, etc.
- No missing values in key fields.
""")

uploaded_file = st.file_uploader("Upload your customer data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📁 Preview of Uploaded Data")
    st.dataframe(df.head())

    required_columns = ['gender', 'Contract', 'PaymentMethod', 'InternetService', 'SeniorCitizen', 'tenure']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        st.stop()

    # Run predictions
    churn_X = preprocess_churn(df)
    tenure_X = preprocess_tenure(df)

    churn_probs = churn_model.predict_proba(churn_X)[:, 1]
    churn_preds = churn_model.predict(churn_X)
    tenure_preds = tenure_model.predict(tenure_X)

    df_result = df.copy()
    df_result["Churn Probability"] = (churn_probs * 100).round(1).astype(str) + "%"
    df_result["Churn Prediction"] = pd.Series(churn_preds).map({0: "No", 1: "Yes"})
    df_result["Predicted Tenure (Months)"] = np.round(tenure_preds, 1)

    st.subheader("📈 Prediction Results")
    st.dataframe(df_result)

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 Filter Options")
    filters = {
        "Contract": st.sidebar.multiselect("Filter by Contract", options=df_result["Contract"].unique()),
        "gender": st.sidebar.multiselect("Filter by Gender", options=df_result["gender"].unique()),
        "InternetService": st.sidebar.multiselect("Filter by Internet Service", options=df_result["InternetService"].unique()),
        "PaymentMethod": st.sidebar.multiselect("Filter by Payment Method", options=df_result["PaymentMethod"].unique()),
        "Churn Prediction": st.sidebar.multiselect("Filter by Churn Prediction", options=df_result["Churn Prediction"].unique()),
    }

    for col, selected in filters.items():
        if selected:
            df_result = df_result[df_result[col].isin(selected)]

    # --- Visualizations ---
    st.subheader("📊 Visual Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Total Churn Distribution**")
        fig1, ax1 = plt.subplots()
        ax1 = sns.countplot(x="Churn Prediction", data=df_result)
        ax1.set_ylabel("Number of Customers")
        ax1.set_title("Total Churn Distribution")
        for p in ax1.patches:
            ax1.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', fontsize=10, color='black', xytext=(0, 10),
                         textcoords='offset points')
        st.pyplot(fig1)

        # Dynamic Insight
        total = len(df_result)
        churned = df_result[df_result["Churn Prediction"] == "Yes"].shape[0]
        churn_rate = (churned / total) * 100 if total else 0
        st.markdown("""
        **Insight:**
        - Total Customers: **{}**  
        - Predicted to Churn: **{}** (**{:.1f}%**)  
        - Most common contract among churners: **{}**  
        - Most common payment method among churners: **{}**
        """.format(
            total,
            churned,
            churn_rate,
            df_result[df_result["Churn Prediction"] == "Yes"]["Contract"].mode()[0] if churned else "N/A",
            df_result[df_result["Churn Prediction"] == "Yes"]["PaymentMethod"].mode()[0] if churned else "N/A"
        ))

    with col2:
        st.markdown("**Total Tenure Prediction**")
        if "tenure" in df_result.columns:
            fig2, ax2 = plt.subplots()
            ax2.scatter(df_result["tenure"], df_result["Predicted Tenure (Months)"], alpha=0.4, label="Predictions")
            ax2.plot([0, 80], [0, 80], 'r--', label="Ideal Fit")

            z = np.polyfit(df_result["tenure"], df_result["Predicted Tenure (Months)"], 1)
            p = np.poly1d(z)
            ax2.plot(df_result["tenure"], p(df_result["tenure"]), "g-", label="Trend Line")

            ax2.set_xlabel("True Tenure (Months)")
            ax2.set_ylabel("Predicted Tenure (Months)")
            ax2.set_title("True vs. Predicted Tenure")
            ax2.legend()
            st.pyplot(fig2)

            r2 = r2_score(df_result["tenure"], df_result["Predicted Tenure (Months)"])
            st.markdown(f"""
            **Insight:**
            This chart compares actual and predicted tenure.
            - The **green line** represents trend fit.
            - The **red dashed line** is perfect prediction.
            - **R² Score:** `{r2:.2f}`

            The model shows {'strong' if r2 > 0.75 else 'moderate' if r2 > 0.5 else 'weak'} correlation between predicted and actual tenure.
            """)

    # --- Download ---
    csv = df_result.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Predictions as CSV", csv, "churn_tenure_predictions.csv", "text/csv")
