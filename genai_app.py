import streamlit as st
import pandas as pd
from services.genai_service import stream_summary_from_dataset, generate_leads_table, detect_user_intent
from shared_data import get_shared_dataset
import concurrent.futures
import time
import pandas as pd

def run():
    st.header("🧠 GenAI-Based Trade Lead Assistant")
    st.caption("Ask the AI assistant about trade opportunities or customer portfolio insights.")

    df = get_shared_dataset()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "output_data" not in st.session_state:
        st.session_state.output_data = []

    # --- Display previous messages ---
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Chat input ---
    user_input = st.chat_input("💬 Ask GenAI about trade opportunities...")

    if user_input:
    # --- Fade out old table (if any) ---
        if (
            "output_data" in st.session_state
            and st.session_state.output_data is not None
            and not (isinstance(st.session_state.output_data, pd.DataFrame) and st.session_state.output_data.empty)
        ):
            fade_css = """
            <style>
            .fade-out {
                opacity: 0;
                transition: opacity 0.6s ease-in-out;
            }
            .fade-slide {
                transform: translateY(10px);
                transition: all 0.6s ease-in-out;
            }
            </style>
            <script>
            const oldTable = window.parent.document.querySelector('div[data-testid="stDataFrame"]');
            const oldHeader = window.parent.document.querySelector('h3');
            if (oldTable) {
                oldTable.classList.add('fade-out', 'fade-slide');
                if (oldHeader) oldHeader.classList.add('fade-out', 'fade-slide');
                setTimeout(() => {
                    if (oldHeader) oldHeader.remove();
                    oldTable.remove();
                }, 600);
            }
            </script>
            """
            st.markdown(fade_css, unsafe_allow_html=True)

            # ✅ Clear previous leads safely, no rerun needed
            st.session_state.output_data = None

        # --- Add user message to chat history ---
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # --- Detect intent first ---
        with st.spinner("Analyzing intent..."):
            intent = detect_user_intent(user_input)

        leads_future = None  # Initialize for async call

        # --- Start lead generation in the background (if needed) ---
        if intent == "lead_generation":
            executor = concurrent.futures.ThreadPoolExecutor()
            leads_future = executor.submit(generate_leads_table, user_input)

        # --- Stream commentary (dataset insights or pre-lead reasoning) ---
        with st.chat_message("ai"):
            response_placeholder = st.empty()
            streamed_text = ""

            try:
                for chunk in stream_summary_from_dataset(user_input, intent=intent):
                    streamed_text += str(chunk or "")
                    response_placeholder.markdown(streamed_text)
            except Exception as e:
                streamed_text = f"Error streaming: {str(e)}"
                response_placeholder.markdown(streamed_text)

            summary_text = streamed_text
            st.session_state.chat_history.append({"role": "ai", "content": summary_text})

        # --- If lead generation was triggered, show animated progress ---
        if leads_future:
            # --- Inject blue progress bar CSS ---
            blue_css = """
            <style>
            div[data-testid="stProgress"] div[role="progressbar"] {
                background-color: #1E90FF;
                height: 1rem;
                transition: background-color 1s ease-in-out;
            }
            </style>
            """
            st.markdown(blue_css, unsafe_allow_html=True)

            st.markdown("### ⚙️ Generating Leads")
            progress_bar = st.progress(0)
            status_placeholder = st.empty()

            # --- Smooth progress updates ---
            for percent_complete in range(0, 100, 10):
                if leads_future.done():
                    break
                progress_bar.progress(percent_complete)
                status_placeholder.info(
                    f"🔄 Identifying potential trade leads... {percent_complete}% complete"
                )
                time.sleep(1.0)

            # --- Wait for result ---
            with st.spinner("Finalizing potential trade leads..."):
                leads_data = leads_future.result(timeout=60)

            # --- Transition bar from blue → green ---
            green_css = """
            <style>
            div[data-testid="stProgress"] div[role="progressbar"] {
                background-color: #32CD32; /* LimeGreen */
                height: 1rem;
                transition: background-color 1s ease-in-out;
            }
            </style>
            """
            st.markdown(green_css, unsafe_allow_html=True)

            progress_bar.progress(100)
            status_placeholder.success("✅ Leads generation complete!")

            # --- Display results ---
            st.session_state.output_data = leads_data

            # ✅ FIX: Safely check DataFrame truth value
            if leads_data is not None and not (isinstance(leads_data, pd.DataFrame) and leads_data.empty):
                st.subheader("📈 Generated Leads")

                leads_df = pd.DataFrame(leads_data)

                # --- Configure wide, scrollable display ---
                st.dataframe(
                    leads_df,
                    use_container_width=True,  # full-width layout
                    height=500,                # fixed height for easier scrolling
                )

                # Optional: apply a clean styling theme
                st.markdown(
                    """
                    <style>
                    div[data-testid="stDataFrame"] table {
                        font-size: 15px;
                        border-collapse: collapse;
                    }
                    div[data-testid="stDataFrame"] thead tr th {
                        background-color: #f0f4f8;
                        color: #222;
                        font-weight: 600;
                        text-align: left;
                    }
                    div[data-testid="stDataFrame"] tbody tr:hover {
                        background-color: #eaf4ff;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning("⚠️ No leads generated or dataset is empty.")

    # --- Insights ---
    # if (
    #     "output_data" in st.session_state
    #     and st.session_state.output_data is not None
    #     and not (isinstance(st.session_state.output_data, pd.DataFrame) and st.session_state.output_data.empty)
    # ):
    #     with st.expander("⚠️ What GenAI Might Miss"):
    #         st.markdown(
    #             """
    #             - **Numerical Depth:** GenAI is qualitative — it can reason creatively but doesn’t handle numeric correlations well.  
    #             - **Consistency:** Output may vary; it generates insights probabilistically.  
    #             - **Validation Needed:** Use ML models to validate data patterns behind AI-suggested leads.
    #             """
    #         )

    st.markdown("---")
    st.caption("💡 Tip: Try asking things like *'Which customers show strong cross-border payment trends?'*")
