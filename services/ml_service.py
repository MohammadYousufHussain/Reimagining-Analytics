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
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)

def train_multiple_models(df, target_col, train_pct, test_pct, progress_callback=None):
    os.makedirs("static/plots", exist_ok=True)

    # ✅ Define columns to exclude from ML model
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    exclude_cols = (
        [f"Payments_{m}" for m in months] +
        [f"Collections_{m}" for m in months] +
        ["High_Volume_Txns","Avg CASA TTM to Sector Median Percentile","Avg Payments TTM to Sector Median Percentile", "Customer Name"]
    )

    # ✅ Clean copy and encode categorical features
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        if col != target_col:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # ✅ Drop excluded columns if present
    drop_cols = [c for c in exclude_cols if c in df.columns]
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    y = df[target_col]
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    # ✅ Handle stratification safely
    stratify_y = y if pd.Series(y).value_counts().min() >= 2 else None
    if stratify_y is None:
        print("⚠️ Not enough samples for stratified split — proceeding without stratification.")

    # ✅ Data splits
    train_frac = train_pct / 100
    test_frac = test_pct / 100
    backtest_frac = 1 - train_frac - test_frac

    X_temp, X_backtest, y_temp, y_backtest = train_test_split(
        X, y, test_size=backtest_frac, random_state=42, stratify=stratify_y
    )
    test_ratio = test_frac / (train_frac + test_frac)
    stratify_temp = y_temp if stratify_y is not None else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio, random_state=42, stratify=stratify_temp
    )

    # ✅ Scaling for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_backtest_scaled = scaler.transform(X_backtest)

    # ✅ Models to train
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier()
    }

    results_summary = []
    model_outputs = {}
    total_models = len(models)
    current = 0

    # ✅ Train models with progress callback
    for name, model in models.items():
        current += 1
        if progress_callback:
            progress_callback(current, total_models, f"Training {name}...")

        X_tr, X_te = (X_train_scaled, X_test_scaled) if name in ["SVM", "KNN", "Logistic Regression"] else (X_train, X_test)
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        prec, rec, f1 = precision_score(y_test, y_pred, zero_division=0), recall_score(y_test, y_pred, zero_division=0), f1_score(y_test, y_pred, zero_division=0)
        results_summary.append({"Model": name, "Precision": prec, "Recall": rec, "F1": f1})
        model_outputs[name] = model

    results_df = pd.DataFrame(results_summary).sort_values(by="F1", ascending=False)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = model_outputs[best_model_name]

    # ✅ Confusion Matrix
    y_pred_best = best_model.predict(X_test if best_model_name not in ["SVM", "KNN", "Logistic Regression"] else X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_best)
    cm_path = "static/plots/confusion_matrix_best.png"
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{best_model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    # ✅ Feature Importance
    fi_path = None
    fi_df = None
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        fi_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(by="Importance", ascending=False).head(20)
        fi_path = "static/plots/feature_importance_best.png"
        plt.figure(figsize=(6, 5))
        sns.barplot(x="Importance", y="Feature", data=fi_df, palette="Blues_d")
        plt.title(f"Top 20 Important Features ({best_model_name})")
        plt.tight_layout()
        plt.savefig(fi_path)
        plt.close()

    # ✅ Out-of-sample backtesting
    X_bt = X_backtest_scaled if best_model_name in ["SVM", "KNN", "Logistic Regression"] else X_backtest
    y_bt_pred = best_model.predict(X_bt)
    y_bt_prob = best_model.predict_proba(X_bt)[:, 1]

    backtest_df = df.iloc[X_backtest.index].copy()
    backtest_df["Predicted Label"] = y_bt_pred
    backtest_df["Predicted Probability"] = y_bt_prob
    backtest_df["Is Trade Customer"] = y_backtest

    if fi_df is not None:
        top_features = fi_df["Feature"].tolist()

        # Base columns for display
        base_cols = [c for c in ["Customer Name", "Predicted Label", "Predicted Probability", "Is Trade Customer"] if c in backtest_df.columns]

        # Combine and remove duplicates while preserving order
        combined_cols = []
        for c in base_cols + [col for col in top_features if col in backtest_df.columns]:
            if c not in combined_cols:
                combined_cols.append(c)

        backtest_df = backtest_df[combined_cols]
    else:
        base_cols = [c for c in ["Customer Name", "Predicted Label", "Predicted Probability", "Is Trade Customer"] if c in backtest_df.columns]
        backtest_df = backtest_df[base_cols]

    bt_prec, bt_rec = precision_score(y_backtest, y_bt_pred, zero_division=0), recall_score(y_backtest, y_bt_pred, zero_division=0)

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

    return results_df, best_model_info
