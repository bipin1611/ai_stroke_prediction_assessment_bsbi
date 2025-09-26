#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stroke Risk Prediction — Practical Skills Assessment
--------------------------------------------------------------------------------------------

"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import json
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Dict, Tuple, List

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import shap

RANDOM_STATE = 42
RESULTS_DIR = "results"


def _read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        # Try CSV by default
        return pd.read_csv(path)


def load_data(data_path: str = None, data_url: str = None) -> pd.DataFrame:
    if data_path and os.path.exists(data_path):
        df = _read_any(data_path)
    elif data_url:
        df = pd.read_csv(data_url)
    else:
        raise FileNotFoundError("Please provide --data_path or --data_url")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_target_column(df: pd.DataFrame, target_hint: str = "stroke") -> str:
    for c in df.columns:
        if str(c).strip().lower() == target_hint.lower():
            return c
    for cand in ["stroke","target","label","outcome","class"]:
        for c in df.columns:
            if str(c).strip().lower() == cand:
                return c
    # heuristic
    for c in df.columns:
        uniques = pd.Series(df[c]).dropna().unique()
        if len(uniques) <= 3:
            lowered = pd.Series([str(u).strip().lower() for u in uniques])
            if set(lowered).issubset({"0","1","yes","no","true","false"}):
                return c
    return ""


def write_log(lines: List[str]):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "log.txt"), "a", encoding="utf-8") as f:
        for ln in lines:
            f.write(str(ln) + "\n")


def basic_eda(df: pd.DataFrame, target: str, make_plots: bool = True) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Diagnostics
    head_vals = df[target].head(10).tolist()
    write_log([
        "=== BASIC EDA DIAGNOSTICS ===",
        f"Target column: {target}",
        f"Columns: {list(df.columns)}",
        f"Target sample values: {head_vals}",
    ])

    # Target distribution
    try:
        s = df[target].astype(str).fillna("NA")
        target_counts = s.value_counts(dropna=False).rename_axis(target).reset_index(name='count')
        target_counts['proportion'] = target_counts['count'] / max(1, target_counts['count'].sum())
        target_counts.to_csv(os.path.join(RESULTS_DIR, "target_distribution.csv"), index=False)

        if make_plots:
            plt.figure()
            target_counts.plot(kind="bar", x=target, y="count", legend=False, title=f"Target distribution ({target})")
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, "target_distribution.png"))
            plt.close()
    except Exception as e:
        write_log([f"[WARN] Target distribution failed: {e}"])

    # Numerical summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        desc.to_csv(os.path.join(RESULTS_DIR, "numeric_summary.csv"))
        if make_plots:
            for col in numeric_cols:
                try:
                    plt.figure()
                    df[col].hist(bins=30)
                    plt.title(f"Histogram: {col}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(RESULTS_DIR, f"hist_{col}.png"))
                    plt.close()

                    plt.figure()
                    df.boxplot(column=col)
                    plt.title(f"Boxplot: {col}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(RESULTS_DIR, f"box_{col}.png"))
                    plt.close()
                except Exception as e:
                    write_log([f"[WARN] Plot for {col} failed: {e}"])
    else:
        write_log(["[INFO] No numeric columns found for summary/plots."])

    # Pearson correlation with target (numeric only)
    try:
        tgt_num = df[target]
        if not np.issubdtype(tgt_num.dtype, np.number):
            mapping = {"yes":1,"no":0,"true":1,"false":0,"1":1,"0":0}
            tgt_num = tgt_num.astype(str).str.strip().str.lower().map(mapping)
        tgt_num = pd.to_numeric(tgt_num, errors="coerce")
        if numeric_cols and tgt_num.notna().any():
            corrs = df[numeric_cols].corrwith(tgt_num, numeric_only=True).sort_values(ascending=False)
            corrs.to_csv(os.path.join(RESULTS_DIR, "pearson_corr_with_target.csv"))
        else:
            write_log(["[INFO] Skipping correlation (no numeric cols or numeric target)."])
    except Exception as e:
        write_log([f"[WARN] Correlation failed: {e}"])


def build_preprocess_pipeline(df: pd.DataFrame, target: str):
    features = [c for c in df.columns if c != target]
    X = df[features]
    y = df[target]

    if not np.issubdtype(y.dtype, np.number):
        y = y.astype(str).str.strip().str.lower().map({"yes":1,"no":0,"true":1,"false":0,"1":1,"0":0})
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor, y


def get_models() -> Dict[str, Pipeline]:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "SVC": SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=15)
    }


def evaluate_model(name: str, model: Pipeline, X_test, y_test, make_plots: bool = True) -> Dict[str, float]:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
    }

    if make_plots:
        try:
            disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
            disp.ax_.set_title(f"Confusion Matrix — {name}")
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{name}.png"))
            plt.close()
        except Exception as e:
            write_log([f"[WARN] Confusion matrix for {name} failed: {e}"])

        try:
            RocCurveDisplay.from_predictions(y_test, y_prob)
            plt.title(f"ROC Curve — {name}")
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"roc_curve_{name}.png"))
            plt.close()
        except Exception as e:
            write_log([f"[WARN] ROC curve for {name} failed: {e}"])

    return metrics


def run_grid_search(name: str, pipeline: Pipeline, X_train, y_train) -> Tuple[Pipeline, Dict]:
    if name == "RandomForest":
        param_grid = {"model__n_estimators": [200, 400], "model__max_depth": [None, 10, 20], "model__min_samples_split": [2, 5]}
    elif name == "LogisticRegression":
        param_grid = {"model__C": [0.1, 1.0, 5.0, 10.0], "model__penalty": ["l2"]}
    elif name == "SVC":
        param_grid = {"model__C": [0.5, 1.0, 2.0], "model__gamma": ["scale", "auto"]}
    else:
        return pipeline, {}

    gs = GridSearchCV(pipeline, param_grid=param_grid, scoring="roc_auc", cv=5, n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def plot_shap_for_tree_model(name: str, fitted_pipeline: Pipeline, X_sample: pd.DataFrame, preprocessor: ColumnTransformer, make_plots: bool = True):
    if not make_plots:
        return
    try:
        model = fitted_pipeline.named_steps["model"]
        X_trans = preprocessor.transform(X_sample)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_trans)
        sv = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values

        feature_names = []
        for name_, trans, cols in preprocessor.transformers_:
            if name_ == "num":
                feature_names.extend(cols)
            elif name_ == "cat":
                try:
                    ohe = trans.named_steps["onehot"]
                    feature_names.extend(ohe.get_feature_names_out(cols))
                except Exception:
                    feature_names.extend(cols)

        shap.summary_plot(sv, X_trans, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"shap_summary_{name}.png"))
        plt.close()
    except Exception as e:
        write_log([f"[WARN] SHAP for {name} failed: {e}"])


def main(args):
    df = load_data(data_path=args.data_path, data_url=args.data_url)
    target_col = args.target.strip() if args.target else "stroke"
    found_col = find_target_column(df, target_col)

    if not found_col:
        print("\n[ERROR] Target column not found.\n")
        print("Columns in your file:", list(df.columns))
        print("Tip 1: Pass --target <YourColumnName>")
        print("Tip 2: Ensure the column is binary (0/1 or yes/no).")
        raise SystemExit(2)
    if found_col != target_col:
        print(f"[INFO] Using detected target column '{found_col}' (instead of '{target_col}').")

    # Reorder with target last (cleaner)
    cols = [c for c in df.columns if c != found_col] + [found_col]
    df = df[cols]

    # EDA
    basic_eda(df, target=found_col, make_plots=not args.no_plots)

    # Data prep & split
    preprocessor, y = build_preprocess_pipeline(df, target=found_col)
    X = df.drop(columns=[found_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = get_models()
    metrics_rows = []
    fitted_models = {}

    for name, estimator in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        fitted_models[name] = pipe
        m = evaluate_model(name, pipe, X_test, y_test, make_plots=not args.no_plots)
        metrics_rows.append({"Model": name, **m})

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="ROC_AUC", ascending=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)
    print("\n=== Base Metrics ===")
    print(metrics_df.to_string(index=False))

    # Tuning top 2
    top_names = metrics_df["Model"].head(2).tolist()
    tuned_rows = []

    for name in top_names:
        print(f"\n[GridSearch] Tuning {name} ...")
        best_estimator, best_params = run_grid_search(
            name,
            Pipeline(steps=[("preprocess", preprocessor), ("model", models[name])]),
            X_train, y_train
        )
        if best_params:
            with open(os.path.join(RESULTS_DIR, f"best_params_{name}.json"), "w") as f:
                json.dump(best_params, f, indent=2)
            tuned_metrics = evaluate_model(f"{name}_Tuned", best_estimator, X_test, y_test, make_plots=not args.no_plots)
            tuned_rows.append({"Model": f"{name}_Tuned", **tuned_metrics})
            fitted_models[f"{name}_Tuned"] = best_estimator

    if tuned_rows:
        tuned_df = pd.DataFrame(tuned_rows)
        full_df = pd.concat([metrics_df, tuned_df], ignore_index=True).sort_values(by="ROC_AUC", ascending=False)
        full_df.to_csv(os.path.join(RESULTS_DIR, "metrics_with_tuned.csv"), index=False)
        print("\n=== Metrics (with tuned) ===")
        print(full_df.to_string(index=False))

    # SHAP for best tree model
    tree_candidates = [k for k in fitted_models if "RandomForest" in k or "DecisionTree" in k]
    if tree_candidates:
        cand_df = pd.read_csv(os.path.join(RESULTS_DIR, "metrics.csv"))
        cand_df = cand_df[cand_df["Model"].isin(tree_candidates)]
        if not cand_df.empty:
            best_tree = cand_df.sort_values(by="ROC_AUC", ascending=False)["Model"].iloc[0]
            best_pipe = fitted_models[best_tree]
            sample = X_test.sample(n=min(200, len(X_test)), random_state=RANDOM_STATE)
            fitted_preprocessor = best_pipe.named_steps["preprocess"]
            plot_shap_for_tree_model(best_tree, best_pipe, sample, fitted_preprocessor, make_plots=not args.no_plots)

    print(f"\nAll artifacts saved to: ./{RESULTS_DIR}\n")
    print("If something failed, open results/log.txt for details.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./healthcare-dataset-stroke-data.csv",
                        help="Path to local CSV/XLSX file")
    parser.add_argument("--data_url", type=str, default="https://raw.githubusercontent.com/abdelDebug/stroke_prediction/master/healthcare-dataset-stroke-data.csv",
                        help="Fallback URL to load CSV if local path not found")
    parser.add_argument("--target", type=str, default="stroke", help="Name of the target column (case-insensitive)")
    parser.add_argument("--no_plots", action="store_true", help="Disable plot generation for troubleshooting")
    args = parser.parse_args()
    main(args)
