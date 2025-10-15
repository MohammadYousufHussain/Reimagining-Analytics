import streamlit as st
import plotly.express as px
import numpy as np
from shared_data import get_shared_dataset

def run():
    st.title("🧭 Data Features - Exploratory Analysis")

    st.write("""
    This dashboard provides an overview of the diverse **data features** available for customer analytics.
    It helps analysts understand the structure, coverage, and variety of attributes across 
    **Customer Accounts**, **Cross Border Transactions**, **Payments & Collections Seasonality**, 
    **Derived Trade Indicators**, and **Counterparty Profiles**.
    """)

    # ==========================================================
    # 1️⃣ Data Selection for Analysis
    # ==========================================================
    st.subheader("🧮 Select Data for Analysis")
    st.write("""
    Choose which dataset subset you'd like to explore — All, Trade, or Non-Trade customers.
    Your selection will be used throughout the dashboard for analysis and visualizations.
    """)

    df = get_shared_dataset()
    options = ["All Data", "Trade Customers Only", "Non-Trade Customers Only"]
    selection = st.selectbox("Select Dataset", options, index=0)

    if selection == "Trade Customers Only":
        df = df[df["Is Trade Customer"] == "Yes"].reset_index(drop=True)
    elif selection == "Non-Trade Customers Only":
        df = df[df["Is Trade Customer"] == "No"].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    st.success(f"✅ Currently displaying {len(df):,} customer records.")

    # ==========================================================
    # 2️⃣ Summary Metrics Panel
    # ==========================================================
    st.markdown("### 📊 Summary Metrics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        avg_balance = np.mean(df["Avg CASA Balance"])
        st.metric("Avg CASA Balance (AED)", f"{avg_balance:,.0f}")
    with col3:
        avg_payments = np.mean(df["Cross Border Payments_TTM"])
        st.metric("Avg Cross Border Payments", f"{avg_payments:,.0f}")
    with col4:
        avg_counterparties = np.mean(df["Number of Counterparties"])
        st.metric("Avg No. of Counterparties", f"{avg_counterparties:,.0f}")

    # ==========================================================
    # 3️⃣ Display Selected Dataset (Full Table)
    # ==========================================================
    st.markdown("### 🧾 Selected Dataset Preview")
    st.write("""
    The table below displays the dataset corresponding to your selection above.  
    You can use this to quickly inspect data values and confirm coverage across key features.
    """)
    st.dataframe(df, use_container_width=True, height=400)

    st.divider()

    # ==========================================================
    # 4️⃣ Customer Accounts Data
    # ==========================================================
    st.subheader("🏦 Customer Accounts Data")
    st.write("""
    These features capture fundamental information about customers — their sectors, business types, 
    and financial behaviors such as account balances and KYC categories.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.histogram(df, x="Sector", title="Customer Distribution by Sector"), use_container_width=True)
    with col2:
        st.plotly_chart(px.histogram(df, x="KYC Turnover Cateogry", title="KYC Turnover Category Distribution"), use_container_width=True)
    
    st.dataframe(df[["Customer Name", "Sector", "KYC Turnover Cateogry", "Trade License Category", "Avg CASA Balance"]].head(10))

    # ==========================================================
    # 5️⃣ Cross Border Transactions Data
    # ==========================================================
    st.subheader("🌍 Cross Border Transactions Data")
    st.write("""
    This section provides insight into customers' international activity through payment and collection flows.
    """)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(px.histogram(df, x="Cross Border Payments_TTM", nbins=20, title="Cross Border Payments Volume"), use_container_width=True)
    with col4:
        st.plotly_chart(px.histogram(df, x="Cross Border Collections_TTM", nbins=20, title="Cross Border Collections Volume"), use_container_width=True)

    st.dataframe(df[["Customer Name", "Cross Border Payments_TTM", "Cross Border Collections_TTM", 
                     "Cross Border Payments Velocity Ratio", "Cross Border Collections Velocity Ratio"]].head(10))

    # ==========================================================
    # 6️⃣ Payments and Collections Seasonality Data
    # ==========================================================
    st.subheader("📈 Payments & Collections Seasonality")
    st.write("""
    Monthly payment and collection data provide a time-based view of customer activity, 
    enabling trend and seasonality analysis.
    """)

    monthly_payment_cols = [c for c in df.columns if c.startswith("Payments_")]
    payments_long = df.melt(id_vars=["Customer Name"], value_vars=monthly_payment_cols, 
                            var_name="Month", value_name="Payment_Amount")
    st.plotly_chart(px.box(payments_long, x="Month", y="Payment_Amount", title="Monthly Payment Seasonality"), use_container_width=True)

    monthly_collection_cols = [c for c in df.columns if c.startswith("Collections_")]
    collections_long = df.melt(id_vars=["Customer Name"], value_vars=monthly_collection_cols, 
                               var_name="Month", value_name="Collection_Amount")
    st.plotly_chart(px.box(collections_long, x="Month", y="Collection_Amount", title="Monthly Collection Seasonality"), use_container_width=True)

    # ==========================================================
    # 7️⃣ Derived Trade Features
    # ==========================================================
    st.subheader("🚢 Derived Trade Features")
    st.write("""
    These variables act as derived indicators that can help infer trade-related behavior 
    such as payments to **shipping companies**, **customs authorities**, **ports**, and **insurance providers**.
    """)

    derived_cols = ["Payment to Ports", "Payment to Shipping Lines", "Payment to Dubai Customs", 
                    "Payment to Marine Insurers", "FX Forward Amount TTM"]
    st.plotly_chart(px.imshow(df[derived_cols].corr(), text_auto=True, 
                              title="Correlation Between Trade-Related Payment Features"), use_container_width=True)
    st.dataframe(df[["Customer Name"] + derived_cols].head(10))

    # ==========================================================
    # 8️⃣ Counterparty Data and Trade Corridors
    # ==========================================================
    st.subheader("🌐 Counterparties and Trade Corridors")
    st.write("""
    This section highlights cross-border counterparties and trade corridor diversity.
    It includes countries customers transact with and new counterparties added recently.
    """)

    counterparty_cols = [c for c in df.columns if "Counterparty_Country" in c]
    top_countries = df[counterparty_cols].melt(value_name="Country").value_counts().reset_index()
    top_countries.columns = ["_", "Country", "Count"]
    st.plotly_chart(px.bar(top_countries.head(15), x="Country", y="Count", title="Top Counterparty Countries"), use_container_width=True)

    st.dataframe(df[["Customer Name", "Number of Counterparties", "Pct of New Overseas Counterparties in last 90 days"]].head(10))

    st.info("✅ Use the dataset selector at the top to filter insights by Trade or Non-Trade customers.")

    # ==========================================================
    # 9️⃣ Call to Action – Select Data Themes for ML Analysis
    # ==========================================================
    st.divider()
    st.subheader("🚀 Prepare for ML Leads Prediction Analysis")

    st.write("""
    In the next section, you can perform **ML-based Leads Prediction** using selected data themes.  
    Choose which **categories of data features** you want to include in your model training.
    """)

    # --- Thematic options mapping ---
    thematic_options = {
        "Customer Accounts Data": {
            "description": "Includes sector, KYC category, CASA balances, and license information.",
            "columns": [
                "Customer Name", "Sector", "KYC Turnover Cateogry", "Trade License Category",
                "Avg CASA Balance", "Avg CASA TTM to Sector Median Percentile"
            ]
        },
        "Cross Border Transactions Data": {
            "description": "Captures payment and collection volumes, and velocity ratios.",
            "columns": [
                "Cross Border Payments_TTM", "Cross Border Collections_TTM",
                "Cross Border Payments Velocity Ratio", "Cross Border Collections Velocity Ratio"
            ]
        },
        "Payments & Collections Seasonality Data": {
            "description": "Adds time-based payment and collection behavior across months.",
            "columns": [c for c in df.columns if c.startswith("Payments_") or c.startswith("Collections_")]
        },
        "Derived Trade Features": {
            "description": "Includes payments to ports, customs, shipping, insurance, and FX forwards.",
            "columns": [
                "Payment to Ports", "Payment to Shipping Lines", "Payment to Dubai Customs",
                "Payment to Marine Insurers", "FX Forward Amount TTM"
            ]
        },
        "Counterparty & Trade Corridor Data": {
            "description": "Covers countries, counterparties, and corridor diversity metrics.",
            "columns": [
                "Number of Counterparties", "Pct of New Overseas Counterparties in last 90 days"
            ] + [c for c in df.columns if "Counterparty_Country" in c]
        }
    }

    # --- Multiselect for themes ---
    selected_themes = st.multiselect(
        "Select Thematic Data Categories to Include for ML Analysis:",
        options=list(thematic_options.keys()),
        default=[

        ],
        help="Select one or more data themes to include as input features for your ML prediction model."
    )

    # --- Display selected themes and their descriptions ---
    if selected_themes:
        st.write("### 🧩 Selected Data Themes for Analysis")
        for theme in selected_themes:
            st.markdown(f"**• {theme}:** {thematic_options[theme]['description']}")

        # --- Combine all selected columns ---
        selected_columns = []
        for theme in selected_themes:
            selected_columns.extend(thematic_options[theme]["columns"])
        selected_columns = list(dict.fromkeys(selected_columns))  # remove duplicates

        # --- Show preview table of selected columns ---
        st.write("### 🔍 Data Preview (Columns Included in Analysis)")
        st.dataframe(df[selected_columns].head(10), use_container_width=True)

        # --- Optional: Store selections for use in ML page ---
        st.session_state["selected_data_themes"] = selected_themes
        st.session_state["selected_columns"] = selected_columns

        st.success(f"✅ {len(selected_columns)} columns selected across {len(selected_themes)} themes. Ready for ML Leads Prediction.")
    else:
        st.warning("Please select at least one data theme to preview associated features.")
