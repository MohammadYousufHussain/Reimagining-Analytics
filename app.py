import streamlit as st
import dashboard_app
import ml_app
import genai_app
import comparison_app  # <-- NEW

st.set_page_config(page_title="SME Trade Activation", layout="wide")

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Data Exploration", "🤖 ML Leads", "🧠 GenAI Leads", "📊 Model Comparison"]
)

if page == "🏠 Data Exploration":
    dashboard_app.run()
elif page == "🤖 ML Leads":
    ml_app.run()
elif page == "🧠 GenAI Leads":
    genai_app.run()
elif page == "📊 Model Comparison":
    comparison_app.run()