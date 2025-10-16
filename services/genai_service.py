import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
from shared_data import get_shared_dataset  # ✅ Import the shared dataset directly
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@st.cache_data(show_spinner=False)
def load_cached_dataset():
    """
    Load and cache the backtest *input* dataset (same as ML’s out-of-sample input).
    Falls back to shared dataset if unavailable.
    """
    from shared_data import shared_state, get_shared_dataset

    # ✅ Prefer the backtest input dataset defined in shared_data.py
    if "df_backtest" in shared_state:
        df_backtest = shared_state["df_backtest"]
        if isinstance(df_backtest, pd.DataFrame) and not df_backtest.empty:
            return df_backtest.copy()

    # ⚙️ Fallback if not yet defined
    st.warning("⚠️ Backtest dataset not found — using shared synthetic dataset instead.")
    return get_shared_dataset()


def detect_user_intent(user_input):
    """
    Simple heuristic + LLM-based hybrid intent detector.
    Returns: 'lead_generation' or 'general'
    """
    # Quick keyword-based check (fast path)
    trigger_words = [ "can be targeted", "which customers"]
    if any(word in user_input.lower() for word in trigger_words):
        return "lead_generation"

    # Fallback: lightweight LLM check (for ambiguous phrasing)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an intent classifier for an AI assistant."},
                {
                    "role": "user",
                    "content": f"Classify this message as either 'lead_generation' or 'general':\n\n{user_input}",
                },
            ],
            temperature=0,
        )
        intent = response.choices[0].message.content.strip().lower()
        if "lead" in intent:
            return "lead_generation"
    except Exception:
        pass

    return "general"



def stream_summary_from_dataset(user_input, intent="general"):
    """Stream dataset-based commentary or targeted analysis."""
    df = load_cached_dataset()
    feature_list = ", ".join(df.columns.astype(str).tolist())

    if intent == "lead_generation":
        system_prompt = (
            "You are an advanced analytics expert specialized in Trade Finance. "
            "The user wants to identify potential trade leads from a dataset. "
            "Provide a brief reasoning summary of what patterns or signals "
            "you might look for before the actual lead generation step. "
            "Keep answer limited to 200 words."
            "Use Markdown for clarity."
        )
    else:
        system_prompt = (
            "You are an advanced analytics expert specializing in SME Trade Finance. "
            "Answer users questions in an insightful manner, provide differntiated insights. "
            "In case if the question is not business development related, politely decline."
            "Keep answer limited to 200 words."
            "Use Markdown headings (##) and bullet points."
        )

    user_prompt = f"""
    Dataset columns:
    {feature_list}

    User message: "{user_input}"
    """

    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            stream=True,
        )

        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                yield token
    except Exception as e:
        yield f"\n\nError streaming summary: {str(e)}"


@st.cache_data(show_spinner=False)
def prepare_leadgen_inputs():
    """
    Prepare and cache the preprocessed dataset and prompt metadata
    used for GenAI lead generation.
    """
    df = load_cached_dataset().copy()

    # --- Column Exclusions (same as ML model) ---
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    exclude_cols = (
        [f"Payments_{m}" for m in months]
        + [f"Collections_{m}" for m in months]
        + [
            "High_Volume_Txns","Avg CASA TTM to Sector Median Percentile",
            "Avg Payments TTM to Sector Median Percentile","Is Trade Customer","Is FX Customer",
            "Is Counterparty of Trade Customer",
            "Payment to Ports","Payment to Shipping Lines","Number of Counterparties",
            "Avg CASA Balance","Avg CASA TTM to Sector Median Percentile",
            "Avg Collections TTM to Sector Median Percentile",
            "Avg Payments TTM to Sector Median Percentile","Payment to Dubai Customs",
            "Payment to Marine Insurers","FX Forward Amount TTM",
            "Cross Border Payments Velocity Ratio","Cross Border Collections Velocity Ratio",
            "Pct of New Overseas Counterparties in last 90 days",
            "Payments for Warehousing or Storage Fee","Increase in Non-AED CASA",
            "Payment to Freight Forwarders","Txns in High Potential Countries","High_Volume_Txns", "Avg_Volume_Txns",
            "Top_1_Collections_Counterparty_Country", "Top_2_Collections_Counterparty_Country",
            "Top_3_Collections_Counterparty_Country", "Top_4_Collections_Counterparty_Country",
            "Top_1_Payments_Counterparty_Country", "Top_2_Payments_Counterparty_Country",
            "Top_3_Payments_Counterparty_Country", "Top_4_Payments_Counterparty_Country","Nature of Business",
        ]
    )

    # ✅ Pre-filter the dataset
    df_filtered = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")

    # ✅ Prepare prompt-friendly text only once
    feature_list = ", ".join(df_filtered.columns.astype(str).tolist())
    # Use small preview to keep prompt compact
    data_preview = df_filtered.head(40).to_markdown(index=False)

    return df_filtered, feature_list, data_preview


def generate_leads_table(user_input, batch_size=20, max_workers=5):
    """
    Concurrent batch version of GenAI-based trade finance lead generation.
    Splits dataset into batches, processes each batch in parallel threads,
    and aggregates the results into one unified DataFrame.
    """
    from shared_data import shared_state

    # ⚡ Use preprocessed cached data
    df_filtered, feature_list, _ = prepare_leadgen_inputs()

    total_rows = len(df_filtered)
    batches = [df_filtered.iloc[i:i+batch_size] for i in range(0, total_rows, batch_size)]
    st.info(f"🧠 Processing {total_rows:,} customers in {len(batches)} batches of {batch_size}...")

    system_prompt = (
        "You are an expert trade relationship manager using AI to identify trade finance leads. "
        "Analyze the dataset and for each customer, assign a 'Predicted Label' "
        "(1 = likely to engage in trade finance, 0 = not likely). "
        "Provide a brief 'rationale' for each prediction. "
        "Return ONLY a valid JSON array with objects containing "
        "'Customer Name', 'Predicted Label', and 'Lead Rationale'. "
        "Do not include markdown, text, or code fences."
    )

    # --- Helper for processing a single batch ---
    def process_batch(batch_idx, batch_df):
        start_time = time.time()
        data_preview = batch_df.to_markdown(index=False)

        user_prompt = f"""
        Dataset columns:
        {feature_list}

        Batch sample ({len(batch_df)} customers):
        {data_preview}

        Task:
        Based on this dataset and the user query ("{user_input}"), classify all customers with Predicted Labels
        and provide rationales.

        Return JSON only.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            structured_text = response.choices[0].message.content.strip()
            leads_json = json.loads(structured_text)
        except Exception as e:
            leads_json = [{"Customer Name": f"Batch {batch_idx} Error", "Lead Rationale": f"Error parsing JSON: {str(e)}", "Predicted Label": 0}]

        duration = time.time() - start_time
        return pd.DataFrame(leads_json), batch_idx, duration

    # --- Run batches concurrently ---
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_batch, i, batch_df): i for i, batch_df in enumerate(batches)}

        progress_bar = st.progress(0)
        for idx, future in enumerate(as_completed(futures)):
            leads_batch, batch_idx, duration = future.result()
            results.append(leads_batch)
            progress_bar.progress((idx + 1) / len(batches))
            st.write(f"✅ Batch {batch_idx + 1}/{len(batches)} completed in {duration:.1f}s")

    progress_bar.progress(1.0)

    # --- Combine all batches ---
    leads_df = pd.concat(results, ignore_index=True)

    # ✅ Normalize column names
    rename_map = {
        "lead_name": "Customer Name",
        "rationale": "Lead Rationale",
        "predicted_label": "Predicted Label",
    }
    leads_df.rename(columns={k: v for k, v in rename_map.items() if k in leads_df.columns}, inplace=True)

    # ✅ Ensure correct dtypes
    if "Predicted Label" in leads_df.columns:
        leads_df["Predicted Label"] = pd.to_numeric(leads_df["Predicted Label"], errors="coerce").fillna(0).astype(int)
    else:
        leads_df["Predicted Label"] = 0

    # ✅ Merge with actuals from backtest
    if "df_backtest" in shared_state and isinstance(shared_state["df_backtest"], pd.DataFrame):
        df_backtest = shared_state["df_backtest"].copy()
        if "Is Trade Customer" in df_backtest.columns:
            df_backtest["Actual Label"] = df_backtest["Is Trade Customer"].map({"Yes": 1, "No": 0})
            leads_df["Customer Name"] = leads_df["Customer Name"].astype(str)
            df_backtest["Customer Name"] = df_backtest["Customer Name"].astype(str)
            leads_df = leads_df.merge(
                df_backtest[["Customer Name", "Actual Label"]],
                on="Customer Name",
                how="left",
            )
        else:
            leads_df["Actual Label"] = None
    else:
        leads_df["Actual Label"] = None

    # ✅ Store for downstream use
    shared_state["genai_results"] = {
        "leads": leads_df,
        "num_leads": len(leads_df),
        "source": "GenAI Lead Generation",
    }

    st.success(f"🎯 Completed {len(batches)} batches in parallel — total leads: {len(leads_df):,}")
    return leads_df

