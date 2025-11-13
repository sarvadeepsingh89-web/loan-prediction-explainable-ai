import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

def clean_raw_dfs(train_df, test_df):
    # copy to avoid side effects
    tr = train_df.copy()
    te = test_df.copy()

    # Dependents: replace "3+" with 3.0 (keep float so NaNs survive)
    for df in (tr, te):
        if "Dependents" in df.columns:
            df["Dependents"] = df["Dependents"].replace("3+", 3).astype(float)

    # Target mapping (only train has target)
    if "Loan_Status" in tr.columns:
        tr["Loan_Status"] = tr["Loan_Status"].map({"Y":1, "N":0})

    # Fixed mappings for simple binary columns
    binary_maps = {
        "Gender": {"Male":1, "Female":0},
        "Married": {"Yes":1, "No":0},
        "Education": {"Graduate":1, "Not Graduate":0},
        "Self_Employed": {"Yes":1, "No":0}
    }
    for col, mapping in binary_maps.items():
        if col in tr.columns:
            tr[col] = tr[col].map(mapping)
        if col in te.columns:
            te[col] = te[col].map(mapping)

    return tr, te


def create_preprocessor():
    robust_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
    minmax_cols = ["Loan_Amount_Term"]
    no_scale_cols = ["Credit_History", "Dependents"]
    cat_cols = ["Property_Area"]

    robust_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", RobustScaler())
    ])

    minmax_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])

    no_scale_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("robust", robust_pipeline, robust_cols),
            ("minmax", minmax_pipeline, minmax_cols),
            ("no_scale", no_scale_pipeline, no_scale_cols),
            ("cat", categorical_pipeline, cat_cols)
        ], remainder="drop"
    )
    return preprocessor


# Try to get feature names from preprocessor
def get_feature_names(preprocessor):
    feature_names = []

    for name, transformer, cols in preprocessor.transformers_:
        # If dropped, skip
        if transformer == "drop":
            continue

        # Handle pipelines (like robust_pipeline, categorical_pipeline, etc.)
        if isinstance(transformer, Pipeline):
            last_step = transformer.steps[-1][1]  # e.g., RobustScaler, OneHotEncoder

            try:
                # If last step has get_feature_names_out (like OHE)
                fn = last_step.get_feature_names_out(cols)
            except AttributeError:
                # Otherwise just use original cols
                fn = cols
        else:
            try:
                fn = transformer.get_feature_names_out(cols)
            except AttributeError:
                fn = cols

        feature_names.extend(fn)

    return feature_names
