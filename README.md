# 🏦 Loan Prediction System with Explainable AI

## 📘 Overview
The **Loan Prediction System** is a machine learning project designed to predict whether a loan application will be **approved or rejected** based on applicant details such as income, credit history, loan amount, and more.  
It also integrates **Explainable AI (XAI)** techniques like **LIME** and **SHAP** to make the model’s decisions transparent and interpretable.

This project demonstrates a complete end-to-end machine learning pipeline — from preprocessing and model training to deployment in a **Streamlit web app** with visual explanations.

---

## 🎯 Business Objective
Financial institutions often need to assess whether a customer is eligible for a loan.  
The objective is to:
- Predict loan approval outcomes accurately.
- Understand **why** the model made a specific prediction.
- Ensure transparency in model decisions for regulatory and business trust.

---

## 🧩 Key Features
- End-to-end ML workflow (data preprocessing → training → deployment)
- Interactive **Streamlit app** for real-time predictions
- **LIME** and **SHAP** integration for explainability
- Modular and reusable Python codebase
- Handles data preprocessing, missing values, and categorical encoding

---

## 🏗️ Project Structure
loan-pred_project/
│
├── data/ # Dataset storage
├── models/ # Saved trained models
├── outputs_lime/ # LIME explanation artifacts
├── outputs_shap/ # SHAP explanation artifacts
├── utils/
│ ├── preprocessing.py # Data preprocessing functions
│ └── explainability.py # SHAP & LIME explanation logic
│
├── app.py # Streamlit web app
├── train_model.py # Model training and saving
├── predict.py # Prediction pipeline
├── run_explainability.py # Script to test explainability
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## ⚙️ Installation and Setup

### 1️⃣ Clone the Repository
git clone https://github.com/sarvadeepsingh89-web/loan-pred_project.git
cd loan-pred_project

2️⃣ Create and Activate Virtual Environment
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On macOS/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit App
streamlit run app.py

🤖 Model Training & Selection
We trained three machine learning models and compared their performance using cross-validation and validation metrics.

📊 Cross-Validation (Training Set)
Model	              Mean F1	Std
Logistic Regression	  0.8699	0.0070
Random Forest	      0.8703	0.0125
XGBoost               0.8723	0.0097

📊 Validation Results
Model	             Validation F1	Accuracy
Logistic Regression	 0.9032	        0.8536
Random Forest	     0.8636	        0.8049
XGBoost (Selected)	 0.9022	        0.8536

Although XGBoost had a slightly lower F1 score compared to Logistic Regression, it was chosen as the final model due to:

Better handling of imbalanced data
Generalization ability on unseen data
Strong real-world robustness

🧠 Explainability (LIME & SHAP)
To make the model interpretable:

SHAP (SHapley Additive exPlanations) visualizes how each feature impacts the prediction.
LIME (Local Interpretable Model-Agnostic Explanations) provides case-specific explanations for a single prediction.

The app generates:
Feature Importance plots
Local explanation visualizations
Textual summaries explaining why a particular decision (approved/rejected) was made.

🖥️ Streamlit Application
The Streamlit web interface allows users to:

Input loan applicant details.
Generate a prediction (Approved / Rejected).
View visual explainability plots instantly after prediction.

🧾 Example Output
Prediction: ✅ Loan Approved
Top Influencing Factors (SHAP):

High Applicant Income (+)
Strong Credit History (+)
High Loan Amount (–)

📦 Dependencies
All dependencies are listed in requirements.txt.
Key libraries:

pandas, numpy
scikit-learn
xgboost
shap, lime
streamlit

🚀 Future Improvements
Include EDA and data visualization notebook.
Integrate advanced hyperparameter tuning (Optuna/Bayesian).
Add model retraining pipeline with new data.

🧑‍💻 Author
Sarvadeep Singh
📧 [sarvadeepsingh89@gmail.com]
📍 Project: Loan Prediction System (Explainable AI + Streamlit)

🏁 Conclusion
This project successfully demonstrates:

How to build an interpretable loan prediction model.
How explainable AI improves transparency and trust in ML decisions.
A complete, deployable Streamlit app ready for real-world use.

