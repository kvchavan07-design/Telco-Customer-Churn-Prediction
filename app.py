import streamlit as st
import pandas as pd
import joblib


# =========================================================
# LOAD MODEL AND FILES
# =========================================================

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Telco Customer Churn",
    page_icon="📊",
    layout="centered"
)


# =========================================================
# TITLE
# =========================================================

st.title("📊 Telco Customer Churn Prediction")

st.write("Enter customer details to predict whether the customer will churn.")


# =========================================================
# CUSTOMER INPUTS
# Default values are set to a HIGH-RISK customer
# =========================================================

gender = st.selectbox(
    "Gender",
    ["Female", "Male"],
    index=0
)

senior = st.selectbox(
    "Senior Citizen",
    [0, 1],
    index=1
)

partner = st.selectbox(
    "Partner",
    ["No", "Yes"],
    index=0
)

dependents = st.selectbox(
    "Dependents",
    ["No", "Yes"],
    index=0
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=1
)

phone = st.selectbox(
    "Phone Service",
    ["No", "Yes"],
    index=1
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"],
    index=0
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"],
    index=1
)

online_security = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"],
    index=0
)

online_backup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"],
    index=0
)

device_protection = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"],
    index=0
)

tech_support = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"],
    index=0
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"],
    index=1
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"],
    index=1
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"],
    index=0
)

paperless = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"],
    index=1
)

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ],
    index=0
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=90.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=90.0
)


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button("🔍 Predict Churn"):

    # -----------------------------------------------------
    # CREATE DATAFRAME
    # -----------------------------------------------------

    customer = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod": [payment],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total]
    })


    # -----------------------------------------------------
    # BINARY ENCODING
    # Same as LabelEncoder in your notebook
    # -----------------------------------------------------

    customer["gender"] = customer["gender"].map({
        "Female": 0,
        "Male": 1
    })

    customer["Partner"] = customer["Partner"].map({
        "No": 0,
        "Yes": 1
    })

    customer["Dependents"] = customer["Dependents"].map({
        "No": 0,
        "Yes": 1
    })

    customer["PhoneService"] = customer["PhoneService"].map({
        "No": 0,
        "Yes": 1
    })


    # -----------------------------------------------------
    # ONE-HOT ENCODING
    # Same as your notebook
    # -----------------------------------------------------

    categorical_columns = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ]

    customer = pd.get_dummies(
        customer,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )


    # -----------------------------------------------------
    # MATCH TRAINING COLUMNS
    # -----------------------------------------------------

    customer = customer.reindex(
        columns=model_columns,
        fill_value=0
    )


    # -----------------------------------------------------
    # SCALE NUMERICAL FEATURES
    # Same as your notebook
    # -----------------------------------------------------

    customer[["MonthlyCharges", "TotalCharges"]] = scaler.transform(
        customer[["MonthlyCharges", "TotalCharges"]]
    )


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(customer)

    probability = model.predict_proba(customer)[0][1]


    # =====================================================
    # RESULT
    # =====================================================

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.error("⚠️ Customer is likely to CHURN")

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

    else:

        st.success("✅ Customer is likely to STAY")

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )