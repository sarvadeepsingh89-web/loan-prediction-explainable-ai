# utils/explainability.py
import os
from pathlib import Path
import traceback
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import pandas as pd
import lime
import lime.lime_tabular
from utils.preprocessing import get_feature_names  # your helper that extracts transformed names

# --- Feature name mapping for readability ---
feature_names_map = {
    "ApplicantIncome": "Income",
    "CoapplicantIncome": "Co-applicant income",
    "LoanAmount": "Loan amount",
    "Loan_Amount_Term": "Loan term duration",
    "Credit_History": "Credit history",
    "Education": "Education level",
    "Married": "Marital status",
    "Self_Employed": "Self-employment status",
    "Dependents": "Number of dependents",
    "Property_Area_Semiurban": "Living in Semiurban area",
    "Property_Area_Urban": "Living in Urban area"
}

def _get_classifier_from_pipeline(pipeline):
    return pipeline.steps[-1][1]

def shap_global_and_local(
    pipeline,
    X_train_raw,
    X_val_raw,
    row_index: int | None = 0,
    top_k: int = 3,
    out_dir: str = "outputs_shap",
    save_plots: bool = True
):
    """
    Robust SHAP helper:
      - saves summary_plot and waterfall image files into out_dir
      - saves a text explanation file (no numeric 'impact' shown)
      - on failure writes an error file (shap_error.txt)
    Returns: (explanation_text_or_error, artifact_dict)
    artifact_dict keys: summary_plot, waterfall_plot, text_explanation, error
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1) get preprocessor & classifier
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
    except Exception as e:
        err = f"Pipeline must have a 'preprocessor' step: {e}"
        (out_path / "shap_error.txt").write_text(err + "\n" + traceback.format_exc())
        return err, {"summary_plot": None, "waterfall_plot": None, "text_explanation": None, "error": str(out_path / "shap_error.txt")}

    classifier = _get_classifier_from_pipeline(pipeline)
    feature_names = get_feature_names(preprocessor)

    # 2) transform data
    try:
        X_train_proc = preprocessor.transform(X_train_raw)
        X_val_proc = preprocessor.transform(X_val_raw)
    except Exception as e:
        (out_path / "shap_error.txt").write_text("Preprocessor transform failed:\n" + traceback.format_exc())
        return "Preprocessor transform failed", {"summary_plot": None, "waterfall_plot": None, "text_explanation": None, "error": str(out_path / "shap_error.txt")}

    # 3) build SHAP explainer
    try:
        explainer = shap.Explainer(classifier, X_train_proc)
        shap_values = explainer(X_val_proc, check_additivity=False)
    except Exception as e:
        (out_path / "shap_error.txt").write_text("SHAP explainer failure:\n" + traceback.format_exc())
        return f"SHAP explainer failure: {e}", {"summary_plot": None, "waterfall_plot": None, "text_explanation": None, "error": str(out_path / "shap_error.txt")}

    # attach feature names (best-effort)
    try:
        shap_values.feature_names = feature_names
    except Exception:
        pass

    artifacts = {"summary_plot": None, "waterfall_plot": None, "text_explanation": None, "error": None}

    # 4) Global summary plot
    try:
        global_path = out_path / "shap_summary.png"
        plt.figure(figsize=(10, 6))
        # pass shap_values; summary_plot usually handles explain objects
        shap.summary_plot(shap_values, X_val_proc, show=False)
        plt.savefig(global_path, bbox_inches="tight")
        plt.close()
        artifacts["summary_plot"] = str(global_path.resolve())
    except Exception as e:
        (out_path / "shap_error.txt").write_text("SHAP summary_plot failed:\n" + traceback.format_exc())
        artifacts["summary_plot"] = None
        artifacts["error"] = str(out_path / "shap_error.txt")

    # If user only wanted global
    if row_index is None:
        return "✅ Global SHAP summary generated.", artifacts

    # 5) Local waterfall — robust handling for SHAP shapes
    try:
        # shap_values[row_index] often works; if not, we handle below
        try:
            row_exp = shap_values[row_index]
        except Exception:
            arr = np.array(shap_values.values)
            # arr could be (n_samples, n_features) or (n_samples, n_classes, n_features)
            if arr.ndim == 2:
                row_vals = arr[row_index]
                base = shap_values.base_values[row_index] if hasattr(shap_values, "base_values") else None
                row_exp = shap.Explanation(values=row_vals, base_values=base, data=X_val_proc[row_index], feature_names=feature_names)
            elif arr.ndim == 3:
                pred = int(pipeline.predict(X_val_raw.iloc[[row_index]])[0])
                row_vals = arr[row_index, pred, :]
                base = shap_values.base_values[row_index, pred] if hasattr(shap_values, "base_values") else None
                row_exp = shap.Explanation(values=row_vals, base_values=base, data=X_val_proc[row_index], feature_names=feature_names)
            else:
                raise RuntimeError("Unhandled SHAP values shape: " + str(arr.shape))

        local_path = out_path / f"shap_waterfall_row{row_index}.png"
        plt.figure(figsize=(8, 6))
        shap.plots.waterfall(row_exp, show=False)
        plt.savefig(local_path, bbox_inches="tight")
        plt.close()
        artifacts["waterfall_plot"] = str(local_path.resolve())
    except Exception as e:
        (out_path / "shap_error.txt").write_text("SHAP waterfall failed:\n" + traceback.format_exc())
        artifacts["waterfall_plot"] = None
        artifacts["error"] = str(out_path / "shap_error.txt")

    # 6) Simple text explanation (no numeric 'impact' printed)
    try:
        row_vals = np.array(row_exp.values)
        feats_and_vals = list(zip(feature_names, row_vals))
        top_feats = sorted(feats_and_vals, key=lambda x: abs(x[1]), reverse=True)[:top_k]
        pred_label = int(pipeline.predict(X_val_raw.iloc[[row_index]])[0])
        label_text = "APPROVED ✅" if pred_label == 1 else "REJECTED ❌"

        lines = []
        for feat, v in top_feats:
            readable = feature_names_map.get(feat, feat)
            if pred_label == 1:
                lines.append(f"- {readable} helped your loan get approved" if v > 0 else f"- {readable} reduced your chances of approval")
            else:
                lines.append(f"- {readable} reduced your chances of approval" if v < 0 else f"- {readable} was positive but not enough for approval")

        explanation_text = f"This loan was {label_text} mainly because:\n" + "\n".join(lines)
        text_file = out_path / f"shap_explanation_row{row_index}.txt"
        text_file.write_text(explanation_text)
        artifacts["text_explanation"] = str(text_file.resolve())
    except Exception:
        artifacts["text_explanation"] = None
        # don't overwrite earlier error file

    return explanation_text, artifacts

# for lime
# utils/explainability.py  (add/replace lime_local_explanation)
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt

def extract_base_feature(lime_feature_str: str):
    for raw in feature_names_map.keys():
        if raw in lime_feature_str:
            return raw
    return lime_feature_str

def lime_local_explanation(
    pipeline,
    X_train_raw,
    X_val_raw,
    row_index: int = 0,
    top_k: int = 3,
    out_dir: str = "outputs_lime",
    save_plots: bool = True
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # preprocessor + classifier
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.steps[-1][1]
    except Exception as e:
        err = f"Pipeline missing steps: {e}"
        (out_path / "lime_error.txt").write_text(err + "\n" + traceback.format_exc())
        return err, {"lime_plot": None, "text_explanation": None, "error": str(out_path / "lime_error.txt")}

    feature_names = get_feature_names(preprocessor)

    # preprocess training/val for LIME
    X_train_proc = preprocessor.transform(X_train_raw)
    X_val_proc = preprocessor.transform(X_val_raw)

    # build explainer (note: LIME expects numeric arrays)
    try:
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train_proc),
            feature_names=feature_names,
            class_names=["Rejected", "Approved"],
            mode="classification"
        )
    except Exception as e:
        (out_path / "lime_error.txt").write_text("LIME setup failed:\n" + traceback.format_exc())
        return f"LIME setup failed: {e}", {"lime_plot": None, "text_explanation": None, "error": str(out_path / "lime_error.txt")}

    # wrapper
    def predict_fn(x):
        # x will be a numpy array in original feature-space (preprocessed already)
        return classifier.predict_proba(x)

    # pick row
    row_proc = preprocessor.transform(X_val_raw.iloc[[row_index]])
    pred_label = int(pipeline.predict(X_val_raw.iloc[[row_index]])[0])
    label = "APPROVED ✅" if pred_label == 1 else "REJECTED ❌"

    try:
        exp = lime_explainer.explain_instance(
            data_row=row_proc[0],
            predict_fn=predict_fn,
            labels=(pred_label,)
        )
    except Exception as e:
        (out_path / "lime_error.txt").write_text("LIME explain_instance failed:\n" + traceback.format_exc())
        return f"LIME explain failure: {e}", {"lime_plot": None, "text_explanation": None, "error": str(out_path / "lime_error.txt")}

    artifacts = {"lime_plot": None, "text_explanation": None, "error": None}

    # save image
    try:
        lime_plot_path = out_path / f"lime_row{row_index}.png"
        fig = exp.as_pyplot_figure(label=pred_label)
        plt.title(f"LIME Explanation for Row {row_index}")
        fig.savefig(lime_plot_path, bbox_inches="tight")
        plt.close(fig)
        artifacts["lime_plot"] = str(lime_plot_path.resolve())
    except Exception as e:
        (out_path / "lime_error.txt").write_text("LIME save image failed:\n" + traceback.format_exc())
        artifacts["lime_plot"] = None
        artifacts["error"] = str(out_path / "lime_error.txt")

    # build human-readable text (no impact numbers)
    try:
        lime_explanation = sorted(exp.as_list(label=pred_label), key=lambda x: abs(x[1]), reverse=True)
        top_reasons = []
        for feature, contribution in lime_explanation[:top_k]:
            clean_feature = extract_base_feature(feature)
            readable_feature = feature_names_map.get(clean_feature, clean_feature)
            if pred_label == 1:
                effect = f"- {readable_feature} helped your loan get approved" if contribution > 0 else f"- {readable_feature} reduced your chances of approval"
            else:
                effect = f"- {readable_feature} reduced your chances of approval" if contribution <= 0 else f"- {readable_feature} was positive but not enough for approval"
            top_reasons.append(effect)

        explanation_text = f"This loan was {label} mainly because:\n" + "\n".join(top_reasons)
        text_file = out_path / f"lime_explanation_row{row_index}.txt"
        text_file.write_text(explanation_text)
        artifacts["text_explanation"] = str(text_file.resolve())
    except Exception:
        artifacts["text_explanation"] = None

    return explanation_text, artifacts
