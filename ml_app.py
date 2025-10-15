import streamlit as st
import pandas as pd
from services.ml_service import train_multiple_models
from shared_data import get_shared_dataset


def run():
    st.title("🤖 Machine Learning Based Lead Identification")
    st.write("Train, compare, and interpret multiple ML models for lead identification.")

    df = get_shared_dataset()
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # 🎯 Target Selection — only boolean columns
    bool_cols = [
        col for col in df.columns
        if df[col].dropna().isin([0, 1, True, False, "Yes", "No"]).all()
    ]
    st.markdown("<h3 style='font-size:22px;'>🎯 Select Target Variable</h3>", unsafe_allow_html=True)
    if bool_cols:
        target_col = st.selectbox("", bool_cols)
    else:
        st.error("No boolean-type columns found for target selection.")
        return

    # ⚙️ Data Split Configuration
    st.subheader("⚙️ Data Split Configuration")
    col1, col2, col3 = st.columns(3)
    train_size = col1.slider("Train %", 40, 80, 50, 5)
    test_size = col2.slider("Test %", 10, 40, 30, 5)
    backtest_size = 100 - train_size - test_size
    col3.markdown(f"**Out-of-sample (Backtest):** {backtest_size}%")

    # 🚀 Run Model Comparison
    if st.button("🚀 Run Model Comparison"):
        progress_bar = st.progress(0)
        progress_text = st.empty()

        def update_progress(current, total, msg):
            progress_bar.progress(current / total)
            progress_text.text(msg)

        with st.spinner("Training models..."):
            results_summary, best_model_info = train_multiple_models(
                df, target_col, train_size, test_size, progress_callback=update_progress
            )

        progress_bar.progress(1.0)
        progress_text.text("✅ All models trained successfully!")
        st.success("✅ Model training and evaluation complete!")

        # 📊 Model Performance Summary
        st.markdown("## 📊 Model Performance Summary")
        left, right = st.columns([2, 1])
        with left:
            st.dataframe(results_summary.style.highlight_max(axis=0, subset=["F1"], color="lightgreen"))
        with right:
            st.markdown("""
            ### ℹ️ Understanding the Metrics
            - **Precision**: Out of all predicted positives, how many are actually correct.  
            - **Recall**: Out of all actual positives, how many were identified correctly.  
            - **F1 Score**: Harmonic mean of Precision and Recall — balances both.
            """)

        # 🏆 Best Model Section
        st.markdown("---")
        st.markdown("## 🏆 Best Model")
        st.markdown(f"### **{best_model_info['model_name']}**")

        # Precision / Recall / F1 in 72px inline HTML
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div style='text-align:center;'>
                <h1 style='font-size:34px; font-weight:800; color:#0f172a; margin-bottom:5px;'>
                    {best_model_info['precision']}
                </h1>
                <p style='font-size:20px; font-weight:600; color:#1e3a8a;'>Precision</p>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div style='text-align:center;'>
                <h1 style='font-size:34px; font-weight:800; color:#0f172a; margin-bottom:5px;'>
                    {best_model_info['recall']}
                </h1>
                <p style='font-size:20px; font-weight:600; color:#1e3a8a;'>Recall</p>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div style='text-align:center;'>
                <h1 style='font-size:34px; font-weight:800; color:#0f172a; margin-bottom:5px;'>
                    {best_model_info['f1']}
                </h1>
                <p style='font-size:20px; font-weight:600; color:#1e3a8a;'>F1 Score</p>
            </div>
            """, unsafe_allow_html=True)

        # Confusion Matrix & Feature Importance side by side
        c1, c2 = st.columns(2)
        with c1:
            st.image(best_model_info["confusion_matrix_plot"], caption="Confusion Matrix")
        with c2:
            if best_model_info["feature_importance_plot"]:
                st.image(best_model_info["feature_importance_plot"], caption="Top 20 Most Important Features")
                st.markdown("""
                ### ℹ️ About Feature Importance
                Feature importance shows how much each variable contributes to reducing model error.
                Higher values mean stronger predictive influence.
                """)

        # 📈 Out-of-Sample Backtesting
        st.markdown("---")
        st.markdown("## 📈 Out-of-Sample Backtesting Results")

        b1, b2 = st.columns(2)
        with b1:
            st.markdown(f"""
            <div style='text-align:center;'>
                <h1 style='font-size:34px; font-weight:800; color:#0f172a; margin-bottom:5px;'>
                    {best_model_info['backtest_precision']}
                </h1>
                <p style='font-size:20px; font-weight:600; color:#1e3a8a;'>Precision (Backtest)</p>
            </div>
            """, unsafe_allow_html=True)
        with b2:
            st.markdown(f"""
            <div style='text-align:center;'>
                <h1 style='font-size:34px; font-weight:800; color:#0f172a; margin-bottom:5px;'>
                    {best_model_info['backtest_recall']}
                </h1>
                <p style='font-size:20px; font-weight:600; color:#1e3a8a;'>Recall (Backtest)</p>
            </div>
            """, unsafe_allow_html=True)

        st.dataframe(best_model_info["backtest_results"])

    else:
        st.info("Click **Run Model Comparison** to start training multiple models.")
