import pandas as pd
import random
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split

# Central state dictionary accessible by all services
shared_state = {}


def get_shared_dataset():
    if "trade_df" not in globals() or "non_trade_df" not in globals():
        sectors = ["Wholesale Trade", "Retail Trade", "Professional Services", "Contracting", "Manufacturing"]
        countries = ["China", "India", "UAE", "Germany", "UK", "KSA","Pakistan","Egypt","Mexico","Taiwan","Singapore","Malaysia","Vietnam"]

        def generate_customer(i, is_trade=True):
            """
            Generate a single synthetic customer record.
            Produces realistic overlap and slight noise between trade and non-trade profiles.
            """

            # --- Core categories ---
            sectors = ["Wholesale Trade", "Retail Trade", "Professional Services", "Contracting", "Manufacturing"]
            countries = ["China", "India", "UAE", "Germany", "UK", "KSA","Pakistan","Egypt","Mexico","Taiwan","Singapore","Malaysia","Vietnam"]

            # --- Basic profile differences between trade and non-trade ---
            if is_trade:
                cb_payments = int(np.random.normal(3_500_000, 800_000))       # Cross Border Payments_TTM
                cb_collections = int(np.random.normal(3_000_000, 700_000))    # Cross Border Collections_TTM
                high_volume_txns = int(np.random.normal(250, 40))             # High_Volume_Txns
                avg_volume_txns = int(np.random.normal(100_000, 20_000))      # Avg_Volume_Txns
                is_fx_customer = random.choices(["Yes", "No"], weights=[0.75, 0.25])[0]  # Is FX Customer
                num_counterparties = int(np.random.normal(70, 15))            # Number of Counterparties
                payment_ports = random.choices(["Yes", "No"], weights=[0.6, 0.4])[0]     # Payment to Ports
                payment_shipping = random.choices(["Yes", "No"], weights=[0.55, 0.45])[0]# Payment to Shipping Lines
                payment_customs = random.choices(["Yes", "No"], weights=[0.55, 0.45])[0] # Payment to Dubai Customs
                license_cat = random.choice(["Commercial", "Industrial", "Professional"]) # Trade License Category
            else:
                cb_payments = int(np.random.normal(3_500_000, 800_000))       # Cross Border Payments_TTM
                cb_collections = int(np.random.normal(3_000_000, 700_000))    # Cross Border Collections_TTM
                high_volume_txns = int(np.random.normal(250, 40))             # High_Volume_Txns
                avg_volume_txns = int(np.random.normal(100_000, 20_000))      # Avg_Volume_Txns
                is_fx_customer = random.choices(["Yes", "No"], weights=[0.75, 0.25])[0]  # Is FX Customer
                num_counterparties = int(np.random.normal(70, 15))            # Number of Counterparties
                payment_ports = random.choices(["Yes", "No"], weights=[0.2, 0.8])[0]     # Payment to Ports
                payment_shipping = random.choices(["Yes", "No"], weights=[0.25, 0.75])[0]# Payment to Shipping Lines
                payment_customs = random.choices(["Yes", "No"], weights=[0.25, 0.75])[0] # Payment to Dubai Customs
                license_cat = random.choice(["Commercial", "Industrial", "Professional"])       # Trade License Category

            # --- Apply small noise to continuous numeric features ---
            cb_payments = max(0, int(cb_payments * np.random.uniform(0.9, 1.1)))
            cb_collections = max(0, int(cb_collections * np.random.uniform(0.9, 1.1)))
            avg_volume_txns = max(0, int(avg_volume_txns * np.random.uniform(0.9, 1.1)))

            # --- Construct row aligned with column_names ---
            row = [
                f"Customer_{i}",                              # Customer Name
                random.choice(sectors),                       # Sector
                random.randint(1, 4),                         # KYC Turnover Category
                "Yes" if is_trade else "No",                  # Is Trade Customer
                is_fx_customer,                               # Is FX Customer
                cb_payments,                                  # Cross Border Payments_TTM
                cb_collections,                               # Cross Border Collections_TTM

                # --- Counterparty countries ---
                *random.choices(countries, k=8),              # Top 4 Collection + 4 Payment Counterparty Countries

                high_volume_txns,                             # High_Volume_Txns
                avg_volume_txns,                              # Avg_Volume_Txns
                random.choice(["Yes", "No"]),                 # Txns in High Potential Countries
                random.randint(2, 10),                        # Large_Txn_Tickets
                random.choice(sectors),                       # Priority_Sectors

                # --- Monthly Payments (12 months) ---
                *[int(np.random.normal(1_200_000, 500_000)) if is_trade else int(np.random.normal(500_000, 250_000)) for _ in range(12)],

                # --- Monthly Collections (12 months) ---
                *[int(np.random.normal(1_000_000, 400_000)) if is_trade else int(np.random.normal(400_000, 200_000)) for _ in range(12)],

                random.choice(["Yes", "No"]),                 # Is Counterparty of Trade Customer
                license_cat,                                  # Trade License Category
                " ".join(random.choices(                      # Nature of Business
                    ["Trading", "Consultancy", "Logistics", "Retail", "Services", "Technology",
                     "Manufacturing", "Energy", "Medical", "Construction"], k=10)
                ),

                int(np.random.normal(60_000, 20_000)) if is_trade else int(np.random.normal(30_000, 15_000)),  # Payment to Ports
                int(np.random.normal(50_000, 15_000)) if is_trade else int(np.random.normal(25_000, 10_000)),  # Payment to Shipping Lines
                num_counterparties,                       # Number of Counterparties
                int(np.random.normal(700_000, 200_000)),  # Avg CASA Balance
                random.randint(45, 65) if is_trade else random.randint(35, 55),  # Avg CASA TTM to Sector Median Percentile
                random.randint(45, 65) if is_trade else random.randint(35, 55),  # Avg Collections TTM to Sector Median Percentile
                random.randint(45, 65) if is_trade else random.randint(35, 55),  # Avg Payments TTM to Sector Median Percentile
                int(np.random.normal(30_000, 15_000)) if is_trade else int(np.random.normal(25_000, 10_000)),  # Payment to Dubai Customs
                int(np.random.normal(30_000, 15_000)) if is_trade else int(np.random.normal(25_000, 10_000)),  # Payment to Marine Insurers
                int(np.random.normal(30_000, 15_000)) if is_trade else int(np.random.normal(25_000, 10_000)),  # FX Forward Amount TTM
                random.randint(40, 100),                  # Cross Border Payments Velocity Ratio
                random.randint(40, 100),                  # Cross Border Collections Velocity Ratio
                random.randint(0, 100),                   # Pct of New Overseas Counterparties in last 90 days
                int(np.random.normal(30_000, 15_000)) if is_trade else int(np.random.normal(25_000, 10_000)),  # Payments for Warehousing or Storage Fee
                int(np.random.normal(200_000, 80_000)),   # Increase in Non-AED CASA
                int(np.random.normal(30_000, 15_000)) if is_trade else int(np.random.normal(25_000, 10_000)),   # Payment to Freight Forwarders
            ]

            return row

        # Columns remain same as your original
        column_names = [
            "Customer Name", "Sector","KYC Turnover Cateogry", "Is Trade Customer", "Is FX Customer",
            "Cross Border Payments_TTM", "Cross Border Collections_TTM",
            "Top_1_Collections_Counterparty_Country", "Top_2_Collections_Counterparty_Country",
            "Top_3_Collections_Counterparty_Country", "Top_4_Collections_Counterparty_Country",
            "Top_1_Payments_Counterparty_Country", "Top_2_Payments_Counterparty_Country",
            "Top_3_Payments_Counterparty_Country", "Top_4_Payments_Counterparty_Country",
            "High_Volume_Txns", "Avg_Volume_Txns", "Txns in High Potential Countries",
            "Large_Txn_Tickets", "Priority_Sectors",
            *[f"Payments_{m}" for m in ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]],
            *[f"Collections_{m}" for m in ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]],
            "Is Counterparty of Trade Customer", "Trade License Category", "Nature of Business",
            "Payment to Ports", "Payment to Shipping Lines","Number of Counterparties","Avg CASA Balance",
            "Avg CASA TTM to Sector Median Percentile","Avg Collections TTM to Sector Median Percentile",
            "Avg Payments TTM to Sector Median Percentile", "Payment to Dubai Customs", "Payment to Marine Insurers",
            "FX Forward Amount TTM","Cross Border Payments Velocity Ratio","Cross Border Collections Velocity Ratio",
            "Pct of New Overseas Counterparties in last 90 days", "Payments for Warehousing or Storage Fee","Increase in Non-AED CASA",
            "Payment to Freight Forwarders"
        ]

        # Generate separate datasets
        trade_data = [generate_customer(i, True) for i in range(1, 1001)]
        non_trade_data = [generate_customer(i, False) for i in range(1001, 2001)]
        

        globals()["trade_df"] = pd.DataFrame(trade_data, columns=column_names)
        globals()["non_trade_df"] = pd.DataFrame(non_trade_data, columns=column_names)
        globals()["comb_df"] = pd.concat([trade_df[:500], non_trade_df[:500],trade_df[500:], non_trade_df[500:]])

        flip_idx = np.random.choice(globals()["comb_df"].index, size=int(0.10 * len(globals()["comb_df"])), replace=False)
        globals()["comb_df"].loc[flip_idx, "Is Trade Customer"] = globals()["comb_df"].loc[flip_idx, "Is Trade Customer"].map({"Yes": "No", "No": "Yes"})


        # ✅ Define Out-of-Sample Backtesting Split (20%)
        df = globals()["comb_df"]
        y = df["Is Trade Customer"].map({"Yes": 1, "No": 0})
        df_train, df_backtest, y_train, y_backtest = train_test_split(
            df, y, test_size=0.2, random_state=42, stratify=y
        )

        # ✅ Limit backtest sample to avoid LLM timeouts
        df_backtest = df_backtest.head(100)
        y_backtest = y_backtest.head(100)

        # ✅ Store in shared_state for global access
        shared_state["df_full"] = df
        shared_state["df_backtest"] = df_backtest
        shared_state["y_backtest"] = y_backtest

    return globals()["comb_df"]
