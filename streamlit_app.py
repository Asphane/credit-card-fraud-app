import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score

# App Config
st.set_page_config(page_title="💳 Fraud Detection App", layout="wide")

st.title("💳 Credit Card Fraud Detection System")

st.markdown("""
Welcome to the interactive **Credit Card Fraud Detector** built using multiple ML models.  
Upload your transaction data to see predictions from **your selected model**.

---

### 🧠 ML Models Supported:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest (best performer)
""")

# Theme toggle
theme = st.sidebar.radio("🎨 Choose Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("<style>body { background-color: #1e1e1e; color: white; }</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>body { background-color: white; color: black; }</style>", unsafe_allow_html=True)

# Sidebar: Model selector
model_choice = st.sidebar.selectbox("📊 Choose a Model", ["Random Forest", "Logistic Regression", "SDGC"])

# Load model
model_paths = {
    "Random Forest": "rf_model.pkl",
    "Logistic Regression": "logistic_model.pkl",
    "SDGC": "sdgc_model.pkl"
}
model = joblib.load(model_paths[model_choice])

st.sidebar.markdown(f"🔧 Model loaded: `{model_choice}`")

uploaded_file = st.file_uploader("📤 Upload your transaction CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    required_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    if not all(col in df.columns for col in required_cols):
        st.error("❌ CSV must contain V1–V28, Amount, and Time columns.")
    else:
        # Standardize Amount and Time
        scaler = StandardScaler()
        df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])

        # Predict
        try:
            probs = model.predict_proba(df)[:, 1]
        except AttributeError:
            st.warning("⚠️ Selected model does not support probability. Showing 0.5 by default.")
            probs = np.full(shape=(len(df),), fill_value=0.5)

        preds = model.predict(df)

        df["Fraud Probability (%)"] = (probs * 100).round(2)
        df["Prediction"] = np.where(preds == 1, "⚠️ Fraud", "✅ Legit")

        # Summary
        total = len(df)
        fraud_count = np.sum(preds)
        fraud_percent = round((fraud_count / total) * 100, 2)

        st.success("✅ Prediction Completed!")
        st.info(f"🔍 Total Transactions: {total}")
        st.warning(f"⚠️ Fraudulent Transactions: {fraud_count}")
        st.success(f"✅ Legitimate Transactions: {total - fraud_count}")
        st.metric("🚨 Fraud Percentage", f"{fraud_percent}%")

        # Display
        output_df = df[["Fraud Probability (%)", "Prediction"] + [col for col in df.columns if col not in ["Fraud Probability (%)", "Prediction"]]]
        st.dataframe(output_df.head(10), height=350)

        csv = output_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download full predictions", csv, "predictions.csv", "text/csv")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Fraud Probability Histogram")
            fig1, ax1 = plt.subplots()
            ax1.hist(probs, bins=50, color='crimson')
            ax1.set_xlabel("Probability of Fraud")
            ax1.set_ylabel("Transaction Count")
            ax1.set_title("Fraud Probability Distribution")
            st.pyplot(fig1)

        with col2:
            st.subheader("🧁 Prediction Summary Pie Chart")
            pie_data = pd.Series(np.where(preds == 1, "Fraud", "Legit")).value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90, colors=["crimson", "green"])
            ax2.axis('equal')
            st.pyplot(fig2)

        # ROC Curve
        if "Class" in df.columns:
            st.subheader("📈 ROC-AUC Curve (if actual 'Class' present)")
            fpr, tpr, _ = roc_curve(df["Class"], probs)
            auc = roc_auc_score(df["Class"], probs)

            fig3, ax3 = plt.subplots()
            ax3.plot(fpr, tpr, color='blue', label=f"AUC = {auc:.4f}")
            ax3.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax3.set_xlabel("False Positive Rate")
            ax3.set_ylabel("True Positive Rate")
            ax3.set_title("ROC Curve")
            ax3.legend()
            st.pyplot(fig3)

            st.metric("🔍 AUC Score", f"{auc:.4f}")
