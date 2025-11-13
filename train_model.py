import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, f1_score, accuracy_score
import joblib

from utils.preprocessing import clean_raw_dfs
from utils.preprocessing import create_preprocessor
from utils.preprocessing import get_feature_names

DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
test_ids = test_df["Loan_ID"].copy()

# ----------------------------
print("Cleaning raw data (deterministic) ...")
train_df, test_df = clean_raw_dfs(train_df, test_df)

# Train / validation split
X = train_df.drop(columns=["Loan_Status"])
y = train_df["Loan_Status"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# create preprocessor 
preprocessor = create_preprocessor()

log_reg_pipeline = Pipeline([
    ("preprocessor", create_preprocessor()),
    ("clf", LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

rf_pipeline = Pipeline([
    ("preprocessor", create_preprocessor()),
    ("clf", RandomForestClassifier(class_weight='balanced', random_state=42))
])

xgb_pipeline = Pipeline([
    ("preprocessor", create_preprocessor()),
    ("clf", XGBClassifier(eval_metric="logloss", random_state=42))
])

# Gridsearchcv

# Logistic Regression
param_grid_lr = {
    "clf__C": [0.01, 0.1, 1, 10, 100],   # Regularization strength
    "clf__penalty": ["l1", "l2"],        # L1 = Lasso, L2 = Ridge
    "clf__solver": ["liblinear", "saga"]
}

# Random Forest
param_grid_rf = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [None, 5, 10],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4]
}

# XGBoost
param_grid_xgb = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [3, 5, 7],
    "clf__learning_rate": [0.01, 0.1, 0.2],
    "clf__subsample": [0.8, 1.0],
    "clf__colsample_bytree": [0.8, 1.0]
}

# Run GridSearchCV for LogisticRegression
gs_lr = GridSearchCV(
    estimator=log_reg_pipeline,
    param_grid=param_grid_lr,
    cv=5,
    scoring="f1"
)
gs_lr.fit(X_train, y_train)
print("LR best params:", gs_lr.best_params_)
print("LR best CV f1:", gs_lr.best_score_)
best_lr = gs_lr.best_estimator_

# Run GridSerachCV for RandomForest
gs_rf = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid_rf,
    cv=5,
    scoring="f1"
)
gs_rf.fit(X_train, y_train)
print("RF best params:", gs_rf.best_params_)
print("RF best CV f1:", gs_rf.best_score_)
best_rf = gs_rf.best_estimator_

gs_xgb = GridSearchCV(
    estimator=xgb_pipeline,
    param_grid=param_grid_xgb,
    cv=5,
    scoring="f1"
)
gs_xgb.fit(X_train, y_train)
print("XGB best params:", gs_xgb.best_params_)
print("XGB best CV f1:", gs_xgb.best_score_)
best_xgb = gs_xgb.best_estimator_

# Cross-validation + Validation evaluation
models = [
    (best_lr, "Logistic Regression"),
    (best_rf, "Random Forest"),
    (best_xgb, "XGBoost")
]

print("\n📊 Cross-validation results (on training set):")
for model, name in models:
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
    print(f"{name}: Mean F1 = {scores.mean():.4f}, Std = {scores.std():.4f}")

# Validation evaluation (hold-out set)
print("\n📊 Validation results (on X_val):")
for model, name in models:
    y_pred = model.predict(X_val)
    print(f"\n{name} Validation F1 = {f1_score(y_val, y_pred):.4f}")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))

# Retrain on full data
X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])

final_xgb = best_xgb.fit(X_full, y_full)
test_pred = final_xgb.predict(test_df)

# save model
MODELS_DIR = Path("models")
model_filename = MODELS_DIR / "loan_approval_model.pkl"
joblib.dump(final_xgb, model_filename)
print(f"✅ Final model saved to: {model_filename}")

# Feature names: get names from fitted preprocessor
feature_names = get_feature_names(final_xgb.named_steps["preprocessor"])
feat_filename = MODELS_DIR / "feature_names.pkl"
joblib.dump(feature_names, feat_filename)
print(f"✅ Feature names saved to: {feat_filename}")

print("\nDone. To create submission, run: python predict.py")
