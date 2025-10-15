import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Helper: Generate Dummy Data
# -------------------------
def generate_dummy_data(seed=42):
    np.random.seed(seed)
    
    features = [
        "Customer Name", "Sector", "KYC Turnover Category", "Is Trade Customer", "Is FX Customer", 
        "Cross Border Payments_TTM", "Cross Border Collections_TTM",
        "Top_1_Collections_Counterparty_Country", "Top_2_Collections_Counterparty_Country",
        "Top_3_Collections_Counterparty_Country", "Top_4_Collections_Counterparty_Country",
        "Top_1_Payments_Counterparty_Country", "Top_2_Payments_Counterparty_Country",
        "Top_3_Payments_Counterparty_Country", "Top_4_Payments_Counterparty_Country",
        "High_Volume_Txns", "Avg_Volume_Txns", "Txns in High Potential Countries",
        "Large_Txn_Tickets", "Priority_Sectors",
        "Is Counterparty of Trade Customer", "Trade License Category", "Nature of Business",
        "Payment to Ports", "Payment to Shipping Lines", "Number of Counterparties",
        "Avg CASA Balance", "Avg CASA TTM to Sector Median Percentile",
        "Avg Collections TTM to Sector Median Percentile", "Avg Payments TTM to Sector Median Percentile",
        "Payment to Dubai Customs", "Payment to Marine Insurers", "FX Forward Amount TTM",
        "Cross Border Payments Velocity Ratio", "Cross Border Collections Velocity Ratio",
        "Pct of New Overseas Counterparties in last 90 days", "Payments for Warehousing or Storage Fee",
        "Increase in Non-AED CASA", "Payment to Freight Forwarders"
    ]

    # Random numeric encoding for simplicity
    df_ml = pd.DataFrame(np.random.rand(100, len(features)), columns=features)
    df_genai = pd.DataFrame(np.random.rand(100, len(features)), columns=features)

    # Simulate conversion labels (ground truth)
    y_true = np.random.randint(0, 2, 100)

    # Force ML to be more precise & recall-strong than GenAI
    y_pred_ml = np.random.binomial(1, 0.75, 100)
    y_pred_genai = np.random.binomial(1, 0.65, 100)

    # Generate lead names & rationales
    ml_leads = pd.DataFrame({
        "Lead Name": [f"ML_Lead_{i+1}" for i in range(100)],
        "Probability Score": np.random.uniform(0.6, 0.95, 100).round(2),
        "Ground Truth": y_true
    })
    
    rationales = [
        "High FX exposure and frequent cross-border payments",
        "Strong trade corridor with high CASA inflow",
        "Rising transaction velocity in priority sector",
        "Increasing counterparty diversification"
    ]
    genai_leads = pd.DataFrame({
        "Lead Name": [f"GenAI_Lead_{i+1}" for i in range(100)],
        "Lead Rationale": np.random.choice(rationales, 100),
        "Ground Truth": y_true
    })
    
    return df_ml, df_genai, ml_leads, genai_leads, y_true, y_pred_ml, y_pred_genai, features


# -------------------------
# Helper: Run Feature Importance
# -------------------------
def calculate_feature_importance(data, y_true, model_type="Random Forest"):
    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y_true)

    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)

    model.fit(data, y_encoded)
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        importance = np.abs(model.coef_[0])

    fi_df = pd.DataFrame({
        "Feature": data.columns,
        "Importance": importance
    }).sort_values("Importance", ascending=False).head(20)
    return fi_df


# -------------------------
# Main Run Function
# -------------------------
def run():
    st.title("📊 ML vs GenAI Backtesting Comparison")

    st.markdown("""
    This page compares **backtesting results** between ML-driven and GenAI-driven leads.
    We evaluate **precision, recall**, and **feature importance**, then explore how
    GenAI can contextualize insights derived from ML models.
    """)

    # Dummy Data
    df_ml, df_genai, ml_leads, genai_leads, y_true, y_pred_ml, y_pred_genai, features = generate_dummy_data()

    # --- Metrics ---
    from sklearn.metrics import precision_score, recall_score
    precision_ml = precision_score(y_true, y_pred_ml)
    recall_ml = recall_score(y_true, y_pred_ml)
    precision_genai = precision_score(y_true, y_pred_genai)
    recall_genai = recall_score(y_true, y_pred_genai)

    diff_precision = precision_genai - precision_ml
    diff_recall = recall_genai - recall_ml

    st.subheader("📈 Backtesting Metrics Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ML Precision", f"{precision_ml:.2f}")
        st.metric("ML Recall", f"{recall_ml:.2f}")
    with col2:
        st.metric("GenAI Precision", f"{precision_genai:.2f}", delta=f"{diff_precision:.2f}")
        st.metric("GenAI Recall", f"{recall_genai:.2f}", delta=f"{diff_recall:.2f}")

    st.markdown("🔎 *As seen above, ML model shows higher precision and recall compared to GenAI model.*")

    # --- Side-by-side tables ---
    st.subheader("📋 Leads Comparison")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### 🤖 ML Model Leads")
        st.dataframe(ml_leads.head(10))
    with col4:
        st.markdown("#### 🧠 GenAI Leads")
        st.dataframe(genai_leads.head(10))

    # --- Feature Importance ---
    st.subheader("🧩 Feature Importance Analysis")

    st.markdown("""
    Choose the model type and dataset to compute feature importance.
    """)

    col5, col6 = st.columns(2)
    with col5:
        model_choice = st.selectbox(
            "Select ML Classifier",
            ["Random Forest", "Gradient Boosting", "Logistic Regression"]
        )
    with col6:
        dataset_choice = st.selectbox(
            "Select Dataset for Feature Importance",
            ["ML Leads", "GenAI Leads"]
        )

    if st.button("🔍 Compute Feature Importance"):
        if dataset_choice == "ML Leads":
            fi_df = calculate_feature_importance(df_ml, y_true, model_choice)
            title = "Top 20 Features (ML Leads)"
        else:
            fi_df = calculate_feature_importance(df_genai, y_true, model_choice)
            title = "Top 20 Features (GenAI Leads)"

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(fi_df["Feature"], fi_df["Importance"])
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title(title)
        plt.gca().invert_yaxis()
        st.pyplot(fig)

    # Always show both side-by-side (dummy plots for layout)
    st.markdown("#### Comparative View of Top Features")
    fi_ml = calculate_feature_importance(df_ml, y_true)
    fi_genai = calculate_feature_importance(df_genai, y_true)
    col7, col8 = st.columns(2)

    with col7:
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ax1.barh(fi_ml["Feature"], fi_ml["Importance"])
        ax1.set_title("ML Model — Top 20 Features")
        plt.gca().invert_yaxis()
        st.pyplot(fig1)

    with col8:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.barh(fi_genai["Feature"], fi_genai["Importance"])
        ax2.set_title("GenAI Model — Top 20 Features")
        plt.gca().invert_yaxis()
        st.pyplot(fig2)

    # --- Contextualized Insights ---
    st.subheader("💡 Contextualized Insights — Combining ML & GenAI Strengths")
    st.markdown("""
    - **ML Models** identify the statistically strongest predictors of conversion (e.g., high *Cross Border Payments_TTM* or *Avg CASA Balance*).
    - **GenAI Models** can **translate these feature signals into contextualized narratives**, such as:
        > “Clients with increasing transaction volumes in high-potential countries and growing FX activity show strong readiness for trade finance engagement.”
    - **Combining both** enables:
        - Precision-driven targeting from ML
        - Narrative-driven engagement from GenAI
        - Feedback loop for feature refinement and insight curation
    """)
    st.success("Together, ML and GenAI enable explainable, human-like decision support for trade activation and lead management.")
