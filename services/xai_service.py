import os
import pandas as pd
import shap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from dotenv import load_dotenv
from openai import OpenAI


# =========================================
# 🔐 OpenAI Client Setup
# =========================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =========================================
# 🧩 Model Training and Prediction
# =========================================
def generate_predictions(df, model_type, train_ratio, max_depth, n_estimators,
                         learning_rate, subsample=0.8, colsample_bytree=0.8,
                         gamma=0.1, reg_lambda=1.0, reg_alpha=0.0, min_child_weight=1):
    """Train model, generate predictions, and compute SHAP explainability."""

    # 1️⃣ Clean & Prepare Dataset
    df = df.copy().reset_index(drop=True)
    df["Is Trade Customer"] = df["Is Trade Customer"].map({"Yes": 1, "No": 0})
    y = df["Is Trade Customer"]

    # 2️⃣ Exclude Irrelevant Columns
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    exclude_cols = (
        [f"Payments_{m}" for m in months] +
        [f"Collections_{m}" for m in months] +
        [
            "High_Volume_Txns",
            "Avg CASA TTM to Sector Median Percentile",
            "Avg Payments TTM to Sector Median Percentile",
            "Customer Name",
            "Lead"
        ]
    )
    drop_cols = [c for c in exclude_cols if c in df.columns]
    X = df.drop(columns=drop_cols + ["Is Trade Customer"], errors="ignore")

    # Encode categorical columns
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # 3️⃣ Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=42, stratify=y
    )

    # 4️⃣ Model Selection
    if model_type == "Random Forest":
        model = RandomForestClassifier(
            max_depth=max_depth, n_estimators=n_estimators,
            random_state=42, min_samples_split=5, min_samples_leaf=2
        )
    elif model_type == "XGBoost":
        model = XGBClassifier(
            max_depth=max_depth, n_estimators=n_estimators,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree, gamma=gamma,
            reg_lambda=reg_lambda, reg_alpha=reg_alpha,
            min_child_weight=min_child_weight,
            random_state=42, eval_metric="logloss", use_label_encoder=False
        )
    else:
        model = LogisticRegression(max_iter=1000, solver="lbfgs")

    # 5️⃣ Train Model
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    predicted_labels = np.where(preds >= 0.5, 1, 0)

    # 6️⃣ SHAP Explainability
    try:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
        shap_df = pd.DataFrame(shap_values.values, columns=X_test.columns)
    except Exception:
        shap_df = pd.DataFrame()

    # 7️⃣ Predictions Output — aligned with SHAP values
    predictions = pd.DataFrame({
        "Customer Name": df.loc[X_test.index, "Customer Name"].values
        if "Customer Name" in df.columns else X_test.index,
        "Predicted Probability": preds,
        "Predicted Label": predicted_labels
    }).reset_index(drop=True)

    # Return all aligned objects
    return {
        "predictions": predictions,
        "shap_values": shap_df,
        "X_test": X_test.reset_index(drop=True),
    }


# =========================================
# 🎨 SHAP Bar Plot for Selected Lead
# =========================================
def plot_lead_shap(lead_name, predictions, shap_values, X_test):
    """Interactive horizontal SHAP bar chart showing all features for a single lead."""
    if shap_values.empty:
        return None

    try:
        lead_idx = predictions.query("`Customer Name` == @lead_name").index[0]
    except IndexError:
        return None

    shap_row = shap_values.loc[lead_idx]
    df_plot = pd.DataFrame({
        "Feature": X_test.columns,
        "SHAP Value": shap_row
    })
    df_plot["abs_value"] = df_plot["SHAP Value"].abs()
    df_plot = df_plot.sort_values(by="abs_value", ascending=True)

    fig = px.bar(
        df_plot,
        x="SHAP Value",
        y="Feature",
        orientation="h",
        color="SHAP Value",
        color_continuous_scale=[
            (0.00, "#0077B6"),  # deep blue
            (0.50, "#00B4D8"),  # bright cyan
            (1.00, "#90E0EF")   # light teal
        ],
        title=f"Feature Impact for {lead_name}",
        height=600
    )

    fig.update_layout(
        xaxis_title="SHAP Value (Impact on Model Prediction)",
        yaxis_title="Feature",
        coloraxis_colorbar_title="Impact",
        template="plotly_white",
        margin=dict(l=60, r=40, t=60, b=40),
    )

    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>SHAP Value: %{x:.4f}<extra></extra>",
        marker_line_color="#FFFFFF",
        marker_line_width=0.5,
    )
    return fig


# =========================================
# 🧠 GenAI Explainability using GPT-4.1
# =========================================
def generate_lead_rationale(lead_name, shap_row, feature_values, predicted_label=None):
    """
    Use OpenAI GPT-4.1 to generate explainability rationale for a given customer.
    The prompt changes dynamically depending on whether the customer was predicted
    as a trade lead or not. The output is structured markdown suitable for business users.
    """

    if shap_row.empty:
        return f"⚠️ No SHAP data available for **{lead_name}**."

    # Select top SHAP features (no absolute values)
    top_features = shap_row.sort_values(ascending=False).head(18)

    # Build list of feature impacts (retain SHAP sign)
    feature_lines = "\n".join([
        f"- **{feat}**: SHAP impact = {val:+.4f}, actual value = {feature_values.get(feat, 'N/A')}"
        for feat, val in top_features.items()
    ])

    # ----------------------------------------
    # Conditional prompts for Lead vs Non-Lead
    # ----------------------------------------
    if str(predicted_label).lower() in ["1", "yes", "lead", "true"]:
        # ✅ Trade Lead prompt
        prompt = f"""
You are a seasoned **banking data analyst** explaining why a customer was predicted
as a **Trade Lead** by an AI model. Your task is to create a clear, structured markdown summary
for relationship managers. Don't quote any numerical values.

Customer: **{lead_name}**

The model has predicted this customer as a likely trade lead based on these SHAP feature impacts
(positive values indicate higher likelihood of trade behavior):

{feature_lines}

Please produce the following structured markdown response:

## Lead Insights

### 📈 Why this Customer was Predicted as a Trade Lead
Explain what makes this customer a likely trade prospect — highlight signals such as
payments to ports or logistics companies, use of FX or remittance products,
linkages with known trade counterparties, or liquidity and transaction patterns
consistent with trading activity.

### 🔍 Key Drivers and Behavioral Insights
Discuss the top features with **positive SHAP influence**, connecting them to trade-related
behaviors like frequent cross-border flows, steady CASA balances, or multiple counterparties.

### 💡 What Trade Specialists Should Explore Further
Provide 2–3 specific suggestions for the Trade Specialist:
- Validate whether FX Forward or cross-border payments are trade-related.
- Check for new counterparties or increased payments to trade corridors.

"""
    else:
        # 🚫 Non-Lead prompt
        prompt = f"""
You are a seasoned **banking data analyst** explaining why a customer was **not**
predicted as a Trade Lead by an AI model. Create a structured markdown summary
for relationship managers.

Customer: **{lead_name}**

The model has determined this customer is unlikely to be a trade lead based on the
following SHAP impacts (negative values decrease trade likelihood):

{feature_lines}

Please provide this markdown response:

## Non-Lead Insights

### 🧭 Why this Customer was Not Predicted as a Trade Lead
Summarize why the customer may lack trade-related indicators — e.g. low or inconsistent
cross-border activity, few counterparties, limited CASA movement, or absence of FX usage.

### ⚙️ Key Influencing Factors
Explain which features most reduced the trade likelihood, describing what business
patterns they may represent (such as domestic-only payments or static balances).

### 🔍 What Trade Specialists Should Monitor
Suggest how RMs can monitor early signs of trade potential:
- Look for trade-license updates or new sectors.
- Watch for emerging counterparties or rising FX volume.
- Identify liquidity changes or payment behavior shifts that could signal trade readiness.
"""

    # ----------------------------------------
    # Call OpenAI GPT-4.1
    # ----------------------------------------
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": (
                    "You are a senior financial data analyst specializing in explainable AI "
                    "for corporate and SME banking. Always respond in professional markdown, "
                    "using concise business-oriented explanations and clear structure."
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=0.55,
            max_tokens=650
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ Unable to generate GenAI rationale:\n\n{e}"
