import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
# Resolve the CSV relative to app.py so this works locally and on Streamlit Cloud.
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

@st.cache_data
def load_data():
    """Load the bundled CSV, or create a deterministic demo dataset if it was not uploaded."""
    required = FEATURES + ["Outcome"]
    if DATA_PATH.is_file():
        df = pd.read_csv(DATA_PATH)
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
        df = df[required].copy()
        df["Outcome"] = df["Outcome"].astype(int)
        return df

    # Fallback prevents deployment from crashing when the data/ folder is omitted from GitHub.
    # These generated rows are synthetic and intended only for software demonstration.
    rng = np.random.default_rng(42)
    n = 1400
    pregnancies = rng.integers(0, 9, n)
    glucose = np.clip(rng.normal(120, 30, n), 55, 220)
    blood_pressure = np.clip(rng.normal(72, 12, n), 40, 120)
    skin_thickness = np.clip(rng.normal(25, 10, n), 5, 60)
    insulin = np.clip(rng.lognormal(np.log(85), 0.55, n), 10, 400)
    bmi = np.clip(rng.normal(30, 7, n), 16, 55)
    pedigree = np.clip(rng.gamma(2, 0.22, n), 0.05, 2.5)
    age = rng.integers(18, 81, n)
    risk_score = (glucose - 110) / 30 + (bmi - 28) / 12 + (age - 40) / 35 + pregnancies / 10 + rng.normal(0, 0.8, n)
    outcome = (risk_score > np.quantile(risk_score, 0.65)).astype(int)
    return pd.DataFrame({
        "Pregnancies": pregnancies, "Glucose": glucose,
        "BloodPressure": blood_pressure, "SkinThickness": skin_thickness,
        "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree, "Age": age,
        "Outcome": outcome
    })

@st.cache_resource
def train_models():
    df = load_data()
    X, y = df[FEATURES], df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=250, min_samples_leaf=3,
                                             class_weight="balanced", random_state=42,
                                             n_jobs=-1))
        ])
    }
    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = model.predict_proba(X_test)[:, 1]
        metrics[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "auc": roc_auc_score(y_test, score),
            "confusion": confusion_matrix(y_test, pred),
            "report": classification_report(y_test, pred, output_dict=True, zero_division=0)
        }
    return models, metrics, len(df), float(y.mean())

st.title("🩺 Diabetes Prediction")
st.write("A machine-learning demonstration that estimates diabetes risk from common health measurements.")
st.warning("Educational demo only: this project uses a **synthetic sample dataset**, not real patient records. Its predictions and scores are not clinically validated and must not be used to diagnose or treat anyone.")
try:
    models, metrics, row_count, positive_rate = train_models()
except Exception as e:
    st.error(f"Could not load or train the model: {e}")
    st.stop()

with st.sidebar:
    st.header("Prediction settings")
    model_name = st.selectbox("Model", list(models.keys()))
    st.caption("Both models are trained automatically from the CSV included in this project.")

left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.subheader("Enter health measurements")
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1, help="Number of pregnancies")
            glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=110.0, step=1.0)
            bp = st.number_input("Blood pressure (mmHg)", min_value=0.0, max_value=200.0, value=72.0, step=1.0)
            skin = st.number_input("Skin thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
        with c2:
            insulin = st.number_input("Insulin (mu U/mL)", min_value=0.0, max_value=1000.0, value=85.0, step=1.0)
            bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=28.0, step=0.1)
            pedigree = st.number_input("Diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.45, step=0.01)
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

with right:
    st.subheader("Model overview")
    a, b = st.columns(2)
    a.metric("Training rows", f"{int(row_count):,}")
    b.metric("Features", len(FEATURES))
    st.write("**Available models**")
    for name, m in metrics.items():
        st.write(f"- {name}: test accuracy **{m['accuracy']:.1%}**, ROC-AUC **{m['auc']:.3f}**")
    st.caption("Metrics are measured on a held-out portion of the synthetic dataset. They do not represent real-world medical performance.")

if submitted:
    values = pd.DataFrame([{
        "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": bp,
        "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree, "Age": age
    }], columns=FEATURES)
    # In the commonly used Pima-style schema, zero can indicate an unavailable measurement.
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        if values.loc[0, col] == 0:
            values.loc[0, col] = np.nan
    model = models[model_name]
    prediction = int(model.predict(values)[0])
    probability = float(model.predict_proba(values)[0, 1])
    st.divider()
    st.subheader("Prediction result")
    r1, r2 = st.columns([1, 1])
    with r1:
        if prediction == 1:
            st.error("Model result: higher predicted likelihood")
        else:
            st.success("Model result: lower predicted likelihood")
        st.metric("Estimated model probability", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
    with r2:
        st.write("**Submitted measurements**")
        st.dataframe(values.rename(columns={"DiabetesPedigreeFunction":"Pedigree function"}).T.rename(columns={0:"Value"}), use_container_width=True)
    st.info("This is a model output, not a diagnosis. If you have concerns about diabetes, consult a qualified healthcare professional and use validated tests such as those recommended by your clinician.")

with st.expander("View test-set evaluation"):
    selected = metrics[model_name]
    st.write(f"**{model_name}**")
    e1, e2 = st.columns(2)
    e1.metric("Accuracy", f"{selected['accuracy']:.1%}")
    e2.metric("ROC-AUC", f"{selected['auc']:.3f}")
    st.write("Confusion matrix (rows = actual, columns = predicted; labels 0 and 1):")
    st.dataframe(pd.DataFrame(selected["confusion"], index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))

st.caption("Built with Streamlit, pandas, NumPy, and scikit-learn. For learning and demonstration only.")
