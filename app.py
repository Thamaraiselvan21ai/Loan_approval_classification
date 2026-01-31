import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Classifier",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🏦 Loan Approval Prediction App")
st.write("Enter applicant details to predict **Loan Approval Status**")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Applicant Details")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
income = st.sidebar.number_input("Annual Income", min_value=1000, value=50000)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=500, value=10000)

loan_reason = st.sidebar.selectbox(
    "Reason for Loan",
    (
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION"
    )
)

credit_score = st.sidebar.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=700
)

# -----------------------------
# Encode Loan Reason
# -----------------------------
loan_reason_mapping = {
    "PERSONAL": 0,
    "EDUCATION": 1,
    "MEDICAL": 2,
    "VENTURE": 3,
    "HOMEIMPROVEMENT": 4,
    "DEBTCONSOLIDATION": 5
}

loan_reason_encoded = loan_reason_mapping[loan_reason]

# -----------------------------
# Dummy Training Data
# (Replace with your real dataset / saved model)
# -----------------------------
X_train = np.array([
    [25, 30000, 5000, 0, 650],
    [35, 60000, 15000, 4, 720],
    [45, 80000, 20000, 5, 780],
    [22, 20000, 7000, 1, 600],
    [50, 90000, 25000, 3, 800],
    [30, 40000, 10000, 2, 680]
])

y_train = np.array([0, 1, 1, 0, 1, 0])  # 0 = Rejected, 1 = Approved

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Loan Status"):
    input_data = np.array([[age, income, loan_amount, loan_reason_encoded, credit_score]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    status = "APPROVED ✅" if prediction == 1 else "REJECTED ❌"

    st.subheader("📌 Loan Status")
    st.success(status)

    st.subheader("📊 Approval Probability")
    st.progress(int(probability * 100))
    st.write(f"Probability of Approval: **{probability:.2%}**")

    st.subheader("📄 Applicant Summary")
    summary_df = pd.DataFrame({
        "Age": [age],
        "Income": [income],
        "Loan Amount": [loan_amount],
        "Loan Reason": [loan_reason],
        "Credit Score": [credit_score]
    })
    st.table(summary_df)

# -----------------------------
# Footer
# -----------------------------
st.caption("🤖 Model: Logistic Regression | Educational Demo")
