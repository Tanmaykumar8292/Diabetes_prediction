# Diabetes Prediction — Machine Learning Project
[diabetesprediction-zy7tcd6qrkrjc2hamqzkd6.streamlit.app](https://diabetesprediction-zy7tcd6qrkrjc2hamqzkd6.streamlit.app/)
A GitHub-ready Streamlit app that trains two scikit-learn classification models and predicts a demo risk category from health measurements.

> **Important limitation:** The included `data/diabetes.csv` is a **synthetic, programmatically generated dataset** for demonstrating an end-to-end machine-learning workflow. It is not a real clinical dataset, has no clinical validation, and must not be used for diagnosis, treatment, or medical decisions. The probability and evaluation scores are illustrative only.

## Features
- Interactive Streamlit user interface
- Logistic Regression and Random Forest models
- Automatic train/test split and training at app startup
- Accuracy, ROC-AUC, and confusion-matrix evaluation
- Median imputation and feature scaling where appropriate
- Local CSV included; no separate dataset download required

## Run locally

Requires Python 3.10–3.12 recommended.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd diabetes_prediction_project
python -m venv .venv
```

Activate the environment:
- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install and launch:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`).

## Push to GitHub

1. Create a new empty repository on GitHub.
2. Extract this project ZIP.
3. Open a terminal inside the extracted folder and run:

```bash
git init
git add .
git commit -m "Add diabetes prediction ML app"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with the URL of your repository.

## Deploy on Streamlit Community Cloud

1. Push the project to GitHub.
2. In Streamlit Community Cloud, create an app from your repository.
3. Set the main file path to `app.py`.
4. Deploy. `requirements.txt` contains the dependencies.

## Dataset schema
`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`, and target `Outcome` (0/1). The included data is synthetic and should be replaced with a properly sourced, permitted, real dataset for any meaningful research evaluation.

## Project structure

```text
diabetes_prediction_project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── diabetes.csv
```
