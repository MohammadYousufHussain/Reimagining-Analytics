import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from shared_data import shared_state

warnings.filterwarnings("ignore", category=UserWarning)

def train_multiple_models(df, target_col, train_pct, test_pct, progress_callback=None):
    os.makedirs("static/plots", exist_ok=True)

    # --- Column Exclusions ---
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    exclude_cols = (
        [f"Payments_{m}" for m in months] +
        [f"Collections_{m}" for m in months] +
        ["High_Volume_Txns","Avg CASA TTM to Sector Median Percentile","Avg Payments TTM to Sector Median Percentile", "Customer Name"]
    )

    # --- Encode Categorical Columns (apply to both df and backtest) ---
    categorical_cols = [col for col in df.select_dtypes(include=["object"]).columns if col != target_col]

    for col in categorical_cols:
        le = LabelEncoder()
        combined_values = pd.concat([df[col].astype(str), shared_state["df_backtest"][col].astype(str)])
        le.fit(combined_values)  # fit on combined to ensure consistent mapping

        df[col] = le.transform(df[col].astype(str))
        shared_state["df_backtest"][col] = le.transform(shared_state["df_backtest"][col].astype(str))

    # --- Feature / Target Separation ---
    drop_cols = [c for c in exclude_cols if c in df.columns]
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    y = df[target_col]

    # ✅ keep as pandas Series with same index
    if y.dtype == "object":
        y = pd.Series(LabelEncoder().fit_transform(y), index=df.index)

    # ✅ Retrieve shared backtest data
    if "df_backtest" not in shared_state or "y_backtest" not in shared_state:
        raise RuntimeError("Backtest data not found in shared_data. Please call get_shared_dataset() first.")

    df_backtest = shared_state["df_backtest"]
    y_backtest = shared_state["y_backtest"]

    # ✅ Use same feature subset and order as training
    X_backtest = df_backtest.reindex(columns=X.columns, fill_value=0)

    # ✅ Exclude backtest samples from train/test pool
    X_remain = X.loc[~X.index.isin(df_backtest.index)]
    y_remain = y.loc[~y.index.isin(df_backtest.index)]

    # --- Train/Test Split ---
    test_ratio = test_pct / (train_pct + test_pct)
    X_train, X_test, y_train, y_test = train_test_split(
        X_remain, y_remain, test_size=test_ratio, random_state=42, stratify=y_remain
    )

    # --- Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_backtest_scaled = scaler.transform(X_backtest)

    # --- Models ---
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier()
    }

    results_summary = []
    model_outputs = {}
    total_models = len(models)

    for i, (name, model) in enumerate(models.items(), 1):
        if progress_callback:
            progress_callback(i, total_models, f"Training {name}...")

        X_tr, X_te = (X_train_scaled, X_test_scaled) if name in ["SVM", "KNN", "Logistic Regression"] else (X_train, X_test)
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        prec, rec, f1 = precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)
        results_summary.append({"Model": name, "Precision": prec, "Recall": rec, "F1": f1})
        model_outputs[name] = model

    results_df = pd.DataFrame(results_summary).sort_values(by="F1", ascending=False)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = model_outputs[best_model_name]

    # --- Confusion Matrix ---
    y_pred_best = best_model.predict(X_test if best_model_name not in ["SVM", "KNN", "Logistic Regression"] else X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_best)
    cm_path = "static/plots/confusion_matrix_best.png"
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{best_model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    # --- Feature Importance ---
    fi_df = None
    fi_path = None
    if hasattr(best_model, "feature_importances_"):
        fi_df = pd.DataFrame({"Feature": X.columns, "Importance": best_model.feature_importances_}).sort_values(by="Importance", ascending=False).head(20)
        fi_path = "static/plots/feature_importance_best.png"
        plt.figure(figsize=(6, 5))
        sns.barplot(x="Importance", y="Feature", data=fi_df, palette="Blues_d")
        plt.title(f"Top 20 Important Features ({best_model_name})")
        plt.tight_layout()
        plt.savefig(fi_path)
        plt.close()

    # ✅ Out-of-Sample Backtesting using shared data
    X_bt = X_backtest_scaled if best_model_name in ["SVM", "KNN", "Logistic Regression"] else X_backtest
    y_bt_pred = best_model.predict(X_bt)
    y_bt_prob = best_model.predict_proba(X_bt)[:, 1]

    backtest_df = df_backtest.copy()
    backtest_df["Predicted Label"] = y_bt_pred
    backtest_df["Predicted Probability"] = y_bt_prob
    backtest_df["Actual Label"] = y_backtest

    bt_prec = precision_score(y_backtest, y_bt_pred)
    bt_rec = recall_score(y_backtest, y_bt_pred)

    best_model_info = {
        "model_name": best_model_name,
        "precision": round(results_df.iloc[0]["Precision"], 3),
        "recall": round(results_df.iloc[0]["Recall"], 3),
        "f1": round(results_df.iloc[0]["F1"], 3),
        "confusion_matrix_plot": cm_path,
        "feature_importance_plot": fi_path,
        "backtest_results": backtest_df.sort_values(by="Predicted Probability", ascending=False).head(20),
        "backtest_precision": round(bt_prec, 3),
        "backtest_recall": round(bt_rec, 3),
    }

    # ✅ Store for reuse by other services
    shared_state["ml_results"] = {
        "summary": results_df,
        "best_model_info": best_model_info,
        "best_model_object": best_model,
    }

    return results_df, best_model_info
