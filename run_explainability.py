# run_explainability.py
import joblib
import pandas as pd
from utils.preprocessing import clean_raw_dfs
from utils import explainability

# Load model
print("🔹 Loading model...")
model = joblib.load("models/loan_approval_model.pkl")

# Load and clean data
print("🔹 Loading and cleaning data...")
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
train_df, test_df = clean_raw_dfs(train_df, test_df)

X = train_df.drop(columns=["Loan_Status"])
y = train_df["Loan_Status"]

# --- GLOBAL SHAP ---
print("🔹 Running GLOBAL SHAP explainability (all data)...")
text_global, artifacts_global = explainability.shap_global_and_local(model, X, X, row_index=None)
print("\nGlobal SHAP explanation complete.\n")

# --- LOCAL SHAP ---
row_index = 5
print(f"🔹 Running LOCAL SHAP for row {row_index}...")
text_local_shap, artifacts_local_shap = explainability.shap_global_and_local(model, X, X, row_index=row_index)
print("\nLocal SHAP explanation:\n", text_local_shap)

# --- LOCAL LIME ---
print(f"\n🔹 Running LOCAL LIME for row {row_index}...")
text_local_lime, artifacts_local_lime = explainability.lime_local_explanation(model, X, X, row_index=row_index)
print("\nLocal LIME explanation:\n", text_local_lime)

print("\n✅ Explainability run complete.")
