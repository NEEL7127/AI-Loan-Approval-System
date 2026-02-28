import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [BASE_DIR / "loan_model.pkl", BASE_DIR.parent / "loan_model.pkl"]

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")


def resolve_model_path() -> Path:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("loan_model.pkl")


@st.cache_resource
def load_artifact(model_path: str):
    return joblib.load(model_path)


def build_input_dataframe():
    with st.form("loan_form"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        applicant_income = st.number_input("Applicant Income", min_value=0, value=5000, step=100)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=1500, step=100)
        loan_amount = st.number_input("Loan Amount", min_value=1, value=150, step=1)
        loan_amount_term = st.number_input("Loan Amount Term", min_value=1, value=360, step=1)
        credit_history = st.selectbox("Credit History", [1, 0])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

        submitted = st.form_submit_button("Predict")

    row = pd.DataFrame(
        [
            {
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_amount_term,
                "Credit_History": credit_history,
                "Property_Area": property_area,
            }
        ]
    )
    return row, submitted


st.title("AI Loan Approval Prediction")
st.caption("Enter applicant details to predict loan approval.")

try:
    model_path = resolve_model_path()
    artifact = load_artifact(str(model_path))
except FileNotFoundError:
    st.error("loan_model.pkl not found. Run loan_model.py first.")
    st.stop()

if not isinstance(artifact, dict) or "model" not in artifact:
    st.error("loan_model.pkl format is outdated. Re-run loan_model.py to regenerate it.")
    st.stop()

model = artifact["model"]
scaler = artifact["scaler"]
encoders = artifact["encoders"]
feature_columns = artifact["feature_columns"]
target_col = artifact["target_column"]

input_row, submitted = build_input_dataframe()

if submitted:
    try:
        encoded_row = input_row.copy()
        for col in ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]:
            encoded_row[col] = encoders[col].transform(encoded_row[col])

        encoded_row = encoded_row[feature_columns]
        scaled_row = scaler.transform(encoded_row)

        pred_encoded = model.predict(scaled_row)[0]
        pred_label = encoders[target_col].inverse_transform([pred_encoded])[0]

        y_encoded = encoders[target_col].transform(["Y"])[0]
        class_index = list(model.classes_).index(y_encoded)
        approval_probability = model.predict_proba(scaled_row)[0][class_index]

        st.metric("Approval Probability", f"{approval_probability * 100:.2f}%")

        if pred_label == "Y":
            st.success("Prediction: Loan Approved")
        else:
            st.error("Prediction: Loan Not Approved")
    except ValueError as exc:
        st.error(f"Prediction failed due to input mismatch: {exc}")
