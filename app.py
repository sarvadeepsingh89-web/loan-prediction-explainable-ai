# app.py
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

from utils import explainability
from utils.preprocessing import clean_raw_dfs

st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="wide")

# -------------------------
# Helpers & cached loaders
# -------------------------
@st.cache_data
def load_train_sample(path="data/train.csv", sample_n=200):
    df = pd.read_csv(path)
    tr, _ = clean_raw_dfs(df, df.copy())
    X = tr.drop(columns=["Loan_Status"])
    return X.sample(min(sample_n, len(X)), random_state=42)

@st.cache_resource
def load_model(path="models/loan_approval_model.pkl"):
    return joblib.load(path)

def _display_image_file(path: str):
    """Open image bytes and show in Streamlit to avoid caching problems."""
    try:
        from PIL import Image
        img = Image.open(path)
        st.image(img, use_container_width=True)
    except Exception as e:
        st.error(f"Could not open image {path}: {e}")

def file_exists(p):
    return p is not None and Path(p).is_file()

# -------------------------
# Load once
# -------------------------
X_train_sample = load_train_sample()
model = load_model()

# Initialize session state store for artifacts / last prediction
st.session_state.setdefault("explain_artifacts", {})       # dict for artifacts by type
st.session_state.setdefault("last_pred", None)              # {"class":..., "proba":...}
st.session_state.setdefault("last_cleaned_input", None)     # cleaned DataFrame row
st.session_state.setdefault("lime_text_shown", False)       # did we show lime text at predict time?
st.session_state.setdefault("artifacts_generated_for", None) # "shap" or "lime" when generated

# -------------------------
# UI: Input form
# -------------------------
st.title("🏦 Loan Approval Prediction App")
st.markdown("Fill in the details below and press **Predict**. LIME textual explanation is shown by default after prediction. Optional visual explainability (SHAP/LIME image) can be generated on demand.")

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])

with col2:
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0, value=2500)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)

with col3:
    loan_amount = st.number_input("Loan Amount", min_value=0, value=100)
    loan_term = st.selectbox("Loan Term (months)", [360, 180, 120, 60])
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# convert dependents string -> numeric
dependents = 3 if dependents == "3+" else int(dependents)

# build single-row raw DataFrame (same columns as training)
input_data = pd.DataFrame([{
    "Gender": gender,
    "Married": married,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Dependents": dependents,
    "Property_Area": property_area
}])

# small optional debug checkbox
if st.checkbox("Show raw input (debug)", value=False):
    st.write(input_data)

# -------------------------
# Predict & basic LIME text (default)
# -------------------------
if st.button("🧮 Predict Loan Approval"):
    # 1) clean input using same deterministic cleaning used in training
    try:
        cleaned_row, _ = clean_raw_dfs(input_data.copy(), input_data.copy())
    except Exception as e:
        st.error(f"Failed to clean input for prediction: {e}")
        raise

    st.session_state["last_cleaned_input"] = cleaned_row

    # 2) model prediction
    try:
        proba = model.predict_proba(cleaned_row)[0][1]
        pred = int(model.predict(cleaned_row)[0])
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        raise

    st.session_state["last_pred"] = {"class": pred, "proba": float(proba)}
    st.session_state["lime_text_shown"] = False
    st.session_state["artifacts_generated_for"] = None

    # 3) show verdict once
    if pred == 1:
        st.success(f"✅ Loan Approved — confidence: {proba * 100:.1f}%")
    else:
        st.error(f"❌ Loan Rejected — confidence: {(1 - proba) * 100:.1f}%")

    # 4) generate default LIME text explanation (quick, small)
    with st.spinner("Generating LIME textual explanation..."):
        try:
            lime_text, lime_artifacts = explainability.lime_local_explanation(
                model,
                X_train_sample,
                cleaned_row,
                row_index=0,
                top_k=3,
                out_dir="outputs_lime",
                save_plots=True
            )
            st.subheader("💬 Why this decision? (LIME)")
            st.markdown(lime_text)

            # store artifacts & text so visualization step can reuse them (no re-run)
            st.session_state["explain_artifacts"]["lime"] = lime_artifacts
            st.session_state["explain_artifacts"]["lime_text"] = lime_text
            st.session_state["lime_text_shown"] = True
        except Exception as e:
            st.warning(f"LIME explanation failed: {e}")
            st.session_state["explain_artifacts"]["lime"] = None
            st.session_state["explain_artifacts"]["lime_text"] = None
            st.session_state["lime_text_shown"] = False

# -------------------------
# Visualization chooser & generation (only visible AFTER a prediction)
# -------------------------
if st.session_state["last_cleaned_input"] is not None:
    st.markdown("### 📊 Model Explainability Options (Advanced)")
    chosen = st.selectbox(
        "Choose a visualization (nothing will be generated until you press Generate):",
        ["None", "SHAP (global + local)", "LIME image & download"]
    )

    if chosen != "None":
        # show Generate button (user must click to create images)
        gen_key = f"gen_{chosen.replace(' ', '_')}"
        if st.button("Generate selected explainability artifacts", key=gen_key):
            cleaned_row = st.session_state["last_cleaned_input"]
            with st.spinner("Generating explainability artifacts (may take a few seconds)..."):
                try:
                    if chosen.startswith("SHAP"):
                        shap_text, artifacts = explainability.shap_global_and_local(
                            model, X_train_sample, cleaned_row, row_index=0, out_dir="outputs_shap", save_plots=True
                        )
                        st.session_state["explain_artifacts"]["shap"] = artifacts
                        st.session_state["explain_artifacts"]["shap_text"] = shap_text
                        st.session_state["artifacts_generated_for"] = "shap"
                        # show text if available
                        if isinstance(shap_text, str):
                            st.write("#### Explanation text (SHAP)")
                            st.markdown(shap_text)
                    else:
                        lime_text, artifacts = explainability.lime_local_explanation(
                            model, X_train_sample, cleaned_row, row_index=0, out_dir="outputs_lime", save_plots=True
                        )
                        st.session_state["explain_artifacts"]["lime"] = artifacts
                        st.session_state["explain_artifacts"]["lime_text"] = lime_text
                        st.session_state["artifacts_generated_for"] = "lime"
                        # if LIME text wasn't shown earlier at predict time, show it now
                        if not st.session_state.get("lime_text_shown") and isinstance(lime_text, str):
                            st.write("#### Explanation text (LIME)")
                            st.markdown(lime_text)
                            st.session_state["lime_text_shown"] = True

                except Exception as e:
                    st.error(f"Artifact generation failed: {e}")

        # After generation attempt (or if previously generated), display artifacts neatly (no raw debug)
        if st.session_state["artifacts_generated_for"] == "shap":
            art = st.session_state["explain_artifacts"].get("shap") or {}
            # Show SHAP global if produced
            if art.get("summary_plot") and Path(art["summary_plot"]).exists():
                st.write("#### 🔹 SHAP Summary (Global)")
                _display_image_file(art["summary_plot"])
            else:
                st.info("SHAP summary not available. Click Generate to create it.")

            # Show SHAP waterfall if produced
            if art.get("waterfall_plot") and Path(art["waterfall_plot"]).exists():
                st.write("#### 🔹 SHAP Waterfall (Local)")
                _display_image_file(art["waterfall_plot"])
            else:
                st.info("SHAP waterfall not available. Click Generate to create it.")

            # show any error log file if explainability reported one
            if art.get("error") and Path(art["error"]).exists():
                st.warning("Explainability produced an error; see log below.")
                st.text(Path(art["error"]).read_text())

        elif st.session_state["artifacts_generated_for"] == "lime":
            art = st.session_state["explain_artifacts"].get("lime") or {}
            if art.get("lime_plot") and Path(art["lime_plot"]).exists():
                st.write("#### 🔹 LIME Visual")
                _display_image_file(art["lime_plot"])
                with open(art["lime_plot"], "rb") as f:
                    st.download_button("⬇️ Download LIME image", data=f, file_name=Path(art["lime_plot"]).name, mime="image/png")
            else:
                st.info("LIME image not available. Click Generate to create it.")

            # download / show saved text if available
            if art.get("text_explanation") and Path(art["text_explanation"]).exists():
                st.write("#### Saved text explanation (LIME)")
                st.text(Path(art["text_explanation"]).read_text())

# Footer
st.markdown("---")
