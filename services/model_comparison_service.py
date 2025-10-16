import pandas as pd
import numpy as np
from shared_data import shared_state
from sklearn.metrics import precision_score, recall_score


# -------------------------
# Fetch ML and GenAI Results
# -------------------------
def get_real_data():
    """
    Fetch ML and GenAI backtesting results from shared_state.
    Returns:
        df_ml, df_genai, ml_leads, genai_leads,
        y_true, y_pred_ml, y_pred_genai, features,
        ml_only_leads, genai_only_leads
    """
    if "ml_results" not in shared_state:
        raise RuntimeError("❌ No ML model results found. Please run the ML training app first.")

    ml_bundle = shared_state["ml_results"]
    ml_results = ml_bundle["best_model_info"]

    # --- ML Data ---
    df_ml = ml_bundle["summary"]
    ml_leads = ml_results["backtest_results"].copy()

    if "Actual Label" not in ml_leads.columns and "Is Trade Customer" in ml_leads.columns:
        ml_leads.rename(columns={"Is Trade Customer": "Actual Label"}, inplace=True)

    ml_leads = ml_leads[
        [c for c in ["Customer Name", "Predicted Probability", "Predicted Label", "Actual Label"]
         if c in ml_leads.columns]
    ]

    y_true = ml_leads["Actual Label"].astype(int)
    y_pred_ml = ml_leads["Predicted Label"].astype(int)
    features = list(df_ml.columns)

    # -------------------------
    # GenAI Data Integration
    # -------------------------
    if "genai_results" in shared_state and isinstance(shared_state["genai_results"], dict):
        genai_bundle = shared_state["genai_results"]
        genai_leads = genai_bundle.get("leads", pd.DataFrame()).copy()

        if not genai_leads.empty:
            expected_cols = ["Customer Name", "Lead Rationale", "Predicted Label", "Actual_Label"]
            for col in expected_cols:
                if col not in genai_leads.columns:
                    genai_leads[col] = None

            genai_leads.rename(columns={"Actual_Label": "Actual Label"}, inplace=True)

            # ✅ Safe numeric conversion
            for col in ["Predicted Label", "Actual Label"]:
                if col in genai_leads.columns:
                    if isinstance(genai_leads[col], (list, tuple, pd.Series, np.ndarray)):
                        genai_leads[col] = pd.to_numeric(genai_leads[col], errors="coerce").fillna(0).astype(int)
                    else:
                        genai_leads[col] = 0
                else:
                    genai_leads[col] = 0

            # ✅ Drop duplicate columns safely
            genai_leads = genai_leads.loc[:, ~genai_leads.columns.duplicated()]

            df_genai = genai_leads.copy()
            y_pred_genai = df_genai["Predicted Label"].astype(int)
        else:
            df_genai, genai_leads, y_pred_genai = _generate_dummy_genai(ml_leads)
    else:
        df_genai, genai_leads, y_pred_genai = _generate_dummy_genai(ml_leads)

    # -------------------------
    # Lead Comparison Section
    # -------------------------
    ml_df = ml_leads[["Customer Name", "Predicted Label", "Actual Label"]].copy()
    ml_df.rename(columns={"Predicted Label": "ML_Predicted"}, inplace=True)

    genai_df = genai_leads[["Customer Name", "Predicted Label"]].copy()
    genai_df.rename(columns={"Predicted Label": "GenAI_Predicted"}, inplace=True)

    # Ensure Customer Name types match before merging
    ml_df["Customer Name"] = ml_df["Customer Name"].astype(str)
    genai_df["Customer Name"] = genai_df["Customer Name"].astype(str)

    # Merge ML and GenAI predictions
    combined = pd.merge(ml_df, genai_df, on="Customer Name", how="outer").fillna(0)
    combined["ML_Predicted"] = combined["ML_Predicted"].astype(int)
    combined["GenAI_Predicted"] = combined["GenAI_Predicted"].astype(int)
    combined["Actual Label"] = combined["Actual Label"].astype(int)

    # ✅ Leads picked by ML but missed by GenAI
    ml_only_leads = combined[
        (combined["ML_Predicted"] == 1) & (combined["GenAI_Predicted"] == 0)
    ].copy()

    # ✅ Leads picked by GenAI but missed by ML
    genai_only_leads = combined[
        (combined["ML_Predicted"] == 0) & (combined["GenAI_Predicted"] == 1)
    ].copy()

    # Add rationale from GenAI output if available
    if "Lead Rationale" in genai_leads.columns:
        ml_only_leads = ml_only_leads.merge(
            genai_leads[["Customer Name", "Lead Rationale"]],
            on="Customer Name",
            how="left"
        )
        genai_only_leads = genai_only_leads.merge(
            genai_leads[["Customer Name", "Lead Rationale"]],
            on="Customer Name",
            how="left"
        )

    # Ensure Actual Label column exists (fallback to 0 if missing)
    for df in [ml_only_leads, genai_only_leads]:
        if "Actual Label" not in df.columns:
            df["Actual Label"] = 0

    # Save comparison results for reuse
    shared_state["lead_comparison"] = {
        "ml_only": ml_only_leads,
        "genai_only": genai_only_leads,
        "combined": combined
    }

    return (
        df_ml,
        df_genai,
        ml_leads,
        genai_leads,
        y_true,
        y_pred_ml,
        y_pred_genai,
        features,
        ml_only_leads,
        genai_only_leads,
    )


# -------------------------
# Dummy GenAI Generator
# -------------------------
def _generate_dummy_genai(ml_leads):
    np.random.seed(42)
    genai_leads = ml_leads.copy()
    genai_leads["Predicted Label"] = np.where(
        np.random.rand(len(ml_leads)) > 0.7,
        1 - genai_leads["Predicted Label"],
        genai_leads["Predicted Label"],
    )
    genai_leads["Lead Rationale"] = np.random.choice(
        [
            "High FX exposure and frequent cross-border payments",
            "Strong trade corridor with high CASA inflow",
            "Rising transaction velocity in priority sector",
            "Increasing counterparty diversification",
            "Consistent non-AED CASA growth across quarters",
        ],
        len(genai_leads),
    )
    df_genai = genai_leads[
        [c for c in ["Customer Name", "Lead Rationale", "Predicted Label", "Actual Label"]
         if c in genai_leads.columns]
    ].copy()
    y_pred_genai = df_genai["Predicted Label"].astype(int)
    return df_genai, genai_leads, y_pred_genai


# -------------------------
# Compute Metrics
# -------------------------
def compute_metrics(y_true, y_pred_ml, y_pred_genai=None):
    """Compute precision and recall for ML and GenAI models, with safe alignment."""
    precision_ml = recall_ml = None

    # --- ML metrics (prefer stored backtest, else recompute)
    if "ml_results" in shared_state:
        best_info = shared_state["ml_results"]["best_model_info"]
        precision_ml = best_info.get("backtest_precision", None)
        recall_ml = best_info.get("backtest_recall", None)

    if precision_ml is None or recall_ml is None:
        precision_ml = precision_score(y_true, y_pred_ml, zero_division=0)
        recall_ml = recall_score(y_true, y_pred_ml, zero_division=0)

    # --- GenAI metrics ---
    precision_genai = recall_genai = np.nan

    if y_pred_genai is not None and len(y_pred_genai) > 0:
        # Align length if GenAI and ML results differ
        min_len = min(len(y_true), len(y_pred_genai))
        y_true_aligned = y_true[:min_len]
        y_pred_genai_aligned = y_pred_genai[:min_len]

        precision_genai = precision_score(y_true_aligned, y_pred_genai_aligned, zero_division=0)
        recall_genai = recall_score(y_true_aligned, y_pred_genai_aligned, zero_division=0)

    return {
        "precision_ml": precision_ml,
        "recall_ml": recall_ml,
        "precision_genai": precision_genai,
        "recall_genai": recall_genai,
        "diff_precision": (
            precision_genai - precision_ml if not np.isnan(precision_genai) else None
        ),
        "diff_recall": (
            recall_genai - recall_ml if not np.isnan(recall_genai) else None
        ),
    }


# -------------------------
# Feature Importance
# -------------------------
def calculate_feature_importance(data=None, y_true=None, model_type=None):
    if "ml_results" not in shared_state:
        raise RuntimeError("❌ ML results not found. Please run training first.")

    best_model = shared_state["ml_results"]["best_model_object"]

    if hasattr(best_model, "feature_importances_"):
        importance = best_model.feature_importances_
        features = best_model.feature_names_in_
    elif hasattr(best_model, "coef_"):
        importance = np.abs(best_model.coef_[0])
        features = best_model.feature_names_in_
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])

    fi_df = pd.DataFrame(
        {"Feature": features, "Importance": importance}
    ).sort_values("Importance", ascending=False).head(20)

    return fi_df
