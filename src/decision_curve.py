
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from .baselines import DEMOGRAPHIC_COLS, LIFESTYLE_COLS, _split_columns
from .models import train_logistic_regression, train_random_forest, train_xgboost

# models
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

# dca net benefit calculation ( tp/n - fp/n * (threshold/(1-threshold)) )
def _net_benefit(y_true: pd.Series, y_prob, threshold: float) -> float:
    preds = y_prob >= threshold
    n = len(y_true)
    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    odds = threshold / (1 - threshold)
    return (tp / n) - (fp / n) * odds

# save the plot short function
def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

# runs the decision curve analysis and saves the results and plot with _save
def run_decision_curve(
    features_path: Path,
    targets_path: Path,
    out_csv: Path,
    out_png: Path,
    feature_set: str = "combined",
) -> pd.DataFrame:
    features = pd.read_csv(features_path)
    targets = pd.read_csv(targets_path)
    df = features.merge(targets, on="SEQN", how="inner")

    # drop the columns that dont survive the preprocessing
    cols = _get_feature_set_cols(feature_set)
    present = [col for col in cols if col in df.columns]
    X = df[present]
    y = df["hypertension"]
    num_cols, cat_cols = _split_columns(present)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25, stratify=y)

    # from 0.05 to 0.8 with step of 0.05
    thresholds = [round(x, 2) for x in [i / 100 for i in range(5, 81, 5)]]
    rows = []

    # add treat all and treat none strategies
    prevalence = y_test.mean()
    for threshold in thresholds:
        odds = threshold / (1 - threshold)
        treat_all = prevalence - (1 - prevalence) * odds
        rows.append({
            "model": "treat_all",
            "feature_set": feature_set,
            "threshold": threshold,
            "net_benefit": treat_all,
        })
        rows.append({
            "model": "treat_none",
            "feature_set": feature_set,
            "threshold": threshold,
            "net_benefit": 0.0,
        })

    for model_name, model_fn in MODEL_FNS.items():
        model = model_fn(X_train, y_train, num_cols, cat_cols)
        y_prob = model.predict_proba(X_test)[:, 1]
        for threshold in thresholds:
            rows.append({
                "model": model_name,
                "feature_set": feature_set,
                "threshold": threshold,
                "net_benefit": _net_benefit(y_test, y_prob, threshold),
            })

    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    # plot the decision curve
    sns.set_style("whitegrid")
    plt.figure(figsize=(9, 6))
    plot_df = out.copy()
    sns.lineplot(data=plot_df, x="threshold", y="net_benefit", hue="model", linewidth=2.2)
    plt.title(f"Decision Curve Analysis ({feature_set})")
    plt.xlabel("Risk Threshold")
    plt.ylabel("Net Benefit")
    plt.xlim(min(thresholds), max(thresholds))
    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    _save(out_png)
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run_decision_curve(
        root / "data" / "processed" / "features.csv",
        root / "data" / "processed" / "targets.csv",
        root / "reports" / "decision_curve.csv",
        root / "figures" / "decision_curve.png",
        feature_set="combined",
    )


if __name__ == "__main__":
    main()
