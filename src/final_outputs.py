
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from .baselines import DEMOGRAPHIC_COLS, LIFESTYLE_COLS, _split_columns
from .models import train_logistic_regression, train_random_forest, train_xgboost
from .recommendations import INTERVENTIONS, apply_intervention

# the three models
MODEL_FNS = {
    "logistic_regression": train_logistic_regression,
    "random_forest": train_random_forest,
    "xgboost": train_xgboost,
}


def _get_feature_set_cols(feature_set: str):
    if feature_set == "demographic_only":
        return DEMOGRAPHIC_COLS
    if feature_set == "lifestyle_only":
        return LIFESTYLE_COLS
    if feature_set == "combined":
        return DEMOGRAPHIC_COLS + LIFESTYLE_COLS
    raise ValueError(f"Unknown feature set: {feature_set}")

# svae results
def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

# pick 20, 50, 80th percentile risk profiles to run sims on
def _select_profiles(X_test: pd.DataFrame, baseline_risk: pd.Series) -> pd.DataFrame:
    quantiles = [0.2, 0.5, 0.8]
    labels = ["lower_risk_profile", "mid_risk_profile", "higher_risk_profile"]
    rows = []
    for q, label in zip(quantiles, labels):
        target = baseline_risk.quantile(q)
        idx = (baseline_risk - target).abs().idxmin()
        rows.append({"profile": label, "index": idx})
    return pd.DataFrame(rows)


def run_profile_charts(features_path: Path, targets_path: Path, baseline_results_path: Path, out_csv: Path, out_png: Path) -> pd.DataFrame:
    features = pd.read_csv(features_path)
    targets = pd.read_csv(targets_path)
    results = pd.read_csv(baseline_results_path)
    eligible = results[results["feature_set"].isin(["lifestyle_only", "combined"])].copy()

    # resuse the best performing model
    best = eligible.sort_values("roc_auc", ascending=False).iloc[0]

    df = features.merge(targets, on="SEQN", how="inner")
    cols = _get_feature_set_cols(best["feature_set"])
    present = [col for col in cols if col in df.columns]
    X = df[present]
    y = df["hypertension"]
    num_cols, cat_cols = _split_columns(present)

    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=25, stratify=y)
    model = MODEL_FNS[best["model"]](X_train, y_train, num_cols, cat_cols)
    baseline_risk = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)

    profiles = _select_profiles(X_test, baseline_risk)
    rows = []
    for _, profile_row in profiles.iterrows(): # run the interventions for each profile
        idx = profile_row["index"]
        label = profile_row["profile"]
        base_row = X_test.loc[[idx]].copy()
        rows.append({
            "profile": label,
            "scenario": "baseline",
            "predicted_risk": baseline_risk.loc[idx],
        })
        for intervention_name, fn in INTERVENTIONS.items():
            sim_row, _ = apply_intervention(intervention_name, base_row)
            sim_risk = model.predict_proba(sim_row)[:, 1][0]
            rows.append({
                "profile": label,
                "scenario": intervention_name,
                "predicted_risk": sim_risk,
            })

    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=out, x="predicted_risk", y="scenario", hue="profile", orient="h")
    plt.title(f"Scenario Comparison for Example Profiles: {best['model']} ({best['feature_set']})")
    plt.xlabel("Predicted Hypertension Risk")
    plt.ylabel("Scenario")
    _save(out_png)
    return out

# create the necessary graphs for each metric
def run_final_comparison_figure(baseline_results_path: Path, out_png: Path) -> pd.DataFrame:
    results = pd.read_csv(baseline_results_path)
    plot_df = results.copy()
    plot_df["label"] = plot_df["model"] + "\n" + plot_df["feature_set"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.barplot(data=plot_df, x="feature_set", y="accuracy", hue="model", ax=axes[0])
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Feature Set")
    axes[0].set_ylabel("Score")

    sns.barplot(data=plot_df, x="feature_set", y="f1", hue="model", ax=axes[1])
    axes[1].set_title("F1")
    axes[1].set_xlabel("Feature Set")
    axes[1].set_ylabel("Score")

    sns.barplot(data=plot_df, x="feature_set", y="roc_auc", hue="model", ax=axes[2])
    axes[2].set_title("ROC AUC")
    axes[2].set_xlabel("Feature Set")
    axes[2].set_ylabel("Score")

    axes[1].legend_.remove()
    axes[2].legend_.remove()
    _save(out_png)
    return results


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    features_path = root / "data" / "processed" / "features.csv"
    targets_path = root / "data" / "processed" / "targets.csv"
    baseline_results_path = root / "reports" / "baseline_results.csv"
    run_profile_charts(
        features_path,
        targets_path,
        baseline_results_path,
        root / "reports" / "profile_scenarios.csv",
        root / "figures" / "profile_scenarios.png",
    )
    run_final_comparison_figure(
        baseline_results_path,
        root / "figures" / "final_model_comparison.png",
    )


if __name__ == "__main__":
    main()
