import streamlit as st
import pandas as pd
import numpy as np
import services.xai_service as xai_service
import shared_data


def run():
    st.title("🔍 Leads Explainability - Leveraging GenAI")

    # =========================================
    # 1️⃣ Load Dataset
    # =========================================
    st.markdown("### 📂 Dataset Overview")

    df = shared_data.get_shared_dataset().copy()
    df["Lead"] = df["Is Trade Customer"].map({"Yes": 1, "No": 0})
    st.dataframe(df.head(10), use_container_width=True)

    # =========================================
    # 2️⃣ Summary Metrics
    # =========================================
    st.markdown("### 📊 Summary Metrics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        st.metric("Avg CASA Balance (AED)", f"{np.mean(df['Avg CASA Balance']):,.0f}")
    with col3:
        st.metric("Avg Cross Border Payments", f"{np.mean(df['Cross Border Payments_TTM']):,.0f}")
    with col4:
        st.metric("Avg No. of Counterparties", f"{np.mean(df['Number of Counterparties']):,.0f}")

    # =========================================
    # 3️⃣ Model Configuration
    # =========================================
    st.divider()
    st.subheader("⚙️ Model Configuration")

    col1, col2, col3 = st.columns(3)
    with col1:
        train_ratio = st.slider("Training Dataset Split", 0.5, 0.9, 0.8)
    with col2:
        model_type = st.selectbox("Algorithm", ["Random Forest", "XGBoost", "Logistic Regression"])
    with col3:
        max_depth = st.slider("Max Depth", 2, 20, 6)

    n_estimators = st.slider("Number of Estimators", 10, 300, 100)
    learning_rate = st.slider("Learning Rate (XGBoost only)", 0.01, 0.5, 0.1)

    # Advanced XGBoost Parameters
    subsample, colsample_bytree, gamma, reg_lambda, reg_alpha, min_child_weight = 0.8, 0.8, 0.1, 1.0, 0.0, 1

    if model_type == "XGBoost":
        st.markdown("#### ⚙️ Advanced XGBoost Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            subsample = st.slider("Subsample", 0.5, 1.0, 0.8, step=0.05)
        with col2:
            colsample_bytree = st.slider("Colsample by Tree", 0.5, 1.0, 0.8, step=0.05)
        with col3:
            gamma = st.slider("Gamma", 0.0, 1.0, 0.1, step=0.05)

        col4, col5, col6 = st.columns(3)
        with col4:
            reg_lambda = st.slider("L2 Regularization (λ)", 0.0, 5.0, 1.0, step=0.1)
        with col5:
            reg_alpha = st.slider("L1 Regularization (α)", 0.0, 5.0, 0.0, step=0.1)
        with col6:
            min_child_weight = st.slider("Min Child Weight", 1, 10, 1)

    st.info(f"Model: {model_type} | Train Ratio: {train_ratio} | Depth: {max_depth}")

    # =========================================
    # 4️⃣ Train Model
    # =========================================
    if st.button("🚀 Train & Generate Leads"):
        with st.spinner("Training model and generating lead predictions..."):
            results = xai_service.generate_predictions(
                df=df,
                model_type=model_type,
                train_ratio=train_ratio,
                max_depth=max_depth,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                reg_lambda=reg_lambda,
                reg_alpha=reg_alpha,
                min_child_weight=min_child_weight
            )
        st.session_state["xai_results"] = results
        st.success("✅ Lead predictions generated successfully!")

    # =========================================
    # 5️⃣ Display Results
    # =========================================
    if "xai_results" in st.session_state:
        results = st.session_state["xai_results"]
        st.subheader("📋 Predicted Leads and Explainability Summary")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Predicted Leads (All Results)**")
            df_pred = results["predictions"]
            st.dataframe(df_pred, use_container_width=True, height=500)
        with col2:
            st.markdown("**SHAP Values (All Customers)**")
            st.dataframe(results["shap_values"], use_container_width=True, height=500)

        # =========================================
        # 6️⃣ Lead-Specific SHAP Visualization
        # =========================================
        st.divider()
        st.subheader("📊 Lead-Specific SHAP Explainability")

        top_preds = results["predictions"]
        shap_df = results["shap_values"]
        X_test = results["X_test"]

        # Filter Toggle
        filter_option = st.radio(
            "Filter Leads by Prediction:",
            ["Show All", "Predicted Leads", "Predicted Non-Leads"],
            horizontal=True
        )

        if filter_option == "Predicted Leads":
            filtered_preds = top_preds[top_preds["Predicted Label"] == 1]
        elif filter_option == "Predicted Non-Leads":
            filtered_preds = top_preds[top_preds["Predicted Label"] == 0]
        else:
            filtered_preds = top_preds

        selected_lead = st.selectbox(
            "Select a Customer to View Explainability",
            options=filtered_preds["Customer Name"].tolist()
        )

        if not shap_df.empty:
            fig = xai_service.plot_lead_shap(selected_lead, top_preds, shap_df, X_test)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Unable to display SHAP values for the selected customer.")
        else:
            st.warning("No SHAP data available for this model.")

        # =========================================
        # 7️⃣ GenAI Explainability Insights
        # =========================================
        st.divider()
        st.subheader("🧠 GenAI Explainability Insights")

        selected_lead_text = st.selectbox(
            "Select a Customer for GenAI Rationale",
            options=filtered_preds["Customer Name"].tolist(),
            key="lead_text_select"
        )

        if st.button("🪄 Generate GenAI Rationale"):
            try:
                lead_idx = top_preds.query("`Customer Name` == @selected_lead_text").index[0]
                shap_row = shap_df.loc[lead_idx]
                feature_values = X_test.loc[lead_idx].to_dict()
                predicted_label = top_preds.loc[lead_idx, "Predicted Label"]

                # ✅ Pass label to xai_service for correct prompt generation
                explanation = xai_service.generate_lead_rationale(
                    lead_name=selected_lead_text,
                    shap_row=shap_row,
                    feature_values=feature_values,
                    predicted_label=predicted_label
                )

                st.markdown(explanation)
            except Exception as e:
                st.error(f"⚠️ Error generating rationale: {e}")
