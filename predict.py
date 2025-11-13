import pandas as pd
import joblib
from pathlib import Path
from utils.preprocessing import clean_raw_dfs   # reuse the same cleaning function

# Paths
DATA_DIR = Path("data")
TEST_PATH = DATA_DIR / "test.csv"

# Load raw test data
print("Loading test data...")
test_df = pd.read_csv(TEST_PATH)
test_ids = test_df["Loan_ID"].copy()

# Clean test data (same as training)
_, test_df = clean_raw_dfs(pd.DataFrame(), test_df)   # pass empty train_df since only test is needed

# Load trained model
print("Loading trained model...")
model = joblib.load("models/loan_approval_model.pkl")

# Predict
print("Predicting on test set...")
test_pred = model.predict(test_df)

# Create submission file
Submission = pd.DataFrame({
    "Loan_ID": test_ids,
    "Loan_Status": test_pred
})
Submission["Loan_Status"] = Submission["Loan_Status"].map({1: "Y", 0: "N"})

# Save
Submission.to_csv(DATA_DIR / "Submission.csv", index=False)
print("✅ Submission file saved at data/Submission.csv")
