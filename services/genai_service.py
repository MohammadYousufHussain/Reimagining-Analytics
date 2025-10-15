import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
from shared_data import get_shared_dataset  # ✅ Import the shared dataset directly

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ✅ Cached dataset load for performance
@st.cache_data(show_spinner=False)
def load_cached_dataset():
    """Load and cache the shared dataset for reuse across GenAI functions."""
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



def generate_leads_table(user_input):
    """Generate structured JSON table of leads using top 100 rows of shared dataset."""
    df = load_cached_dataset().head(50)  # ✅ Use only top 100 rows
    feature_list = ", ".join(df.columns.astype(str).tolist())

    # Convert top 100 rows to a compact text table
    data_preview = df.to_markdown(index=False)

    system_prompt = (
        "You are an expert trade relationship manager using AI to identify trade finance leads. "
        "Analyze the following sample dataset and suggest customers likely to deepen trade activity. "
        "Return ONLY a valid JSON array of objects with 'lead_name' and 'rationale'. "
        "Do not include markdown, text, or code fences."
    )

    user_prompt = f"""
    Dataset columns:
    {feature_list}

    Dataset sample (top 100 rows):
    {data_preview}

    Task:
    Based on this dataset and the user query ("{user_input}"), identify potential leads for trade activation.

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
        return json.loads(structured_text)
    except Exception as e:
        return [{"lead_name": "Error", "rationale": f"Error parsing JSON: {str(e)}"}]
