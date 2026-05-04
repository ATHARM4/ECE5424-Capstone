
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from .baselines import DEMOGRAPHIC_COLS, LIFESTYLE_COLS, _split_columns
from .models import train_logistic_regression, train_random_forest, train_xgboost


MODEL_FNS = {
    "logistic_regression": train_logistic_regression,
    "random_forest": train_random_forest,
    "xgboost": train_xgboost,
}

# map feature set name to columns
def _get_feature_set_cols(feature_set: str):
    if feature_set == "demographic_only":
        return DEMOGRAPHIC_COLS
    if feature_set == "lifestyle_only":
        return LIFESTYLE_COLS
    if feature_set == "combined":
        return DEMOGRAPHIC_COLS + LIFESTYLE_COLS
    raise ValueError(f"Unknown feature set: {feature_set}")

# change names for better interpretation
def _collapse_feature_name(name: str) -> str:
    if name.startswith("num__"):
        return name.replace("num__", "", 1)
    if name.startswith("cat__"):
        remainder = name.replace("cat__", "", 1)
        if "_" in remainder:
            return remainder.split("_", 1)[0]
        return remainder
    return name

# get the importance + sort it
def _extract_importance(model) -> pd.DataFrame:
    preprocess = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    feature_names = preprocess.get_feature_names_out()

    if hasattr(estimator, "coef_"):
        raw_values = estimator.coef_[0]
        importance = abs(raw_values)
    else:
        raw_values = estimator.feature_importances_
        importance = raw_values

    importance_df = pd.DataFrame({
        "encoded_feature": feature_names,
        "feature": [_collapse_feature_name(name) for name in feature_names],
        "importance": importance,
        "raw_value": raw_values,
    })
    importance_df = importance_df.groupby("feature", as_index=False).agg({
        "importance": "sum",
        "raw_value": "sum",
    })
    importance_df = importance_df.sort_values("importance", ascending=False)
    return importance_df


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

# ru the interpretation model
def run_interpretation(features_path: Path, targets_path: Path, baseline_results_path: Path, out_csv: Path, out_png: Path) -> pd.DataFrame:
    features = pd.read_csv(features_path)
    targets = pd.read_csv(targets_path)
    results = pd.read_csv(baseline_results_path)
    best = results.sort_values("roc_auc", ascending=False).iloc[0]

    df = features.merge(targets, on="SEQN", how="inner")
    cols = _get_feature_set_cols(best["feature_set"])
    present = [col for col in cols if col in df.columns]
    X = df[present]
    y = df["hypertension"]
    num_cols, cat_cols = _split_columns(present)

    # use the best performing model to get feature importance
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=25, stratify=y)
    model = MODEL_FNS[best["model"]](X_train, y_train, num_cols, cat_cols)
    importance_df = _extract_importance(model)
    importance_df["model"] = best["model"]
    importance_df["feature_set"] = best["feature_set"]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(out_csv, index=False)

    # sort and plot x amount of figures (15, initally tbd)
    top = importance_df.head(15).sort_values("importance", ascending=True)
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    sns.barplot(data=top, x="importance", y="feature", orient="h")
    plt.title(f"Top Features: {best['model']} ({best['feature_set']})")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    _save(out_png)
    return importance_df


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    features_path = root / "data" / "processed" / "features.csv"
    targets_path = root / "data" / "processed" / "targets.csv"
    baseline_results_path = root / "reports" / "baseline_results.csv"
    out_csv = root / "reports" / "feature_importances.csv"
    out_png = root / "figures" / "feature_importances_top15.png"
    run_interpretation(features_path, targets_path, baseline_results_path, out_csv, out_png)


if __name__ == "__main__":
    main()
