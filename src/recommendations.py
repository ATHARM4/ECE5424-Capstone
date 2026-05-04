
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


def _get_feature_set_cols(feature_set: str):
    if feature_set == "demographic_only":
        return DEMOGRAPHIC_COLS
    if feature_set == "lifestyle_only":
        return LIFESTYLE_COLS
    if feature_set == "combined":
        return DEMOGRAPHIC_COLS + LIFESTYLE_COLS
    raise ValueError(f"Unknown feature set: {feature_set}")

# condense best into another fn
def _best_lifestyle_row(results: pd.DataFrame) -> pd.Series:
    eligible = results[results["feature_set"].isin(["lifestyle_only", "combined"])].copy()
    return eligible.sort_values("roc_auc", ascending=False).iloc[0]

# intervention - reduce sodium
def _reduce_sodium(X: pd.DataFrame, pct: float):
    X_new = X.copy()
    eligible = pd.Series(False, index=X_new.index)
    if "diet_sodi" in X_new.columns:
        eligible = X_new["diet_sodi"].notna() & (X_new["diet_sodi"] > 2300)
        X_new.loc[eligible, "diet_sodi"] = X_new.loc[eligible, "diet_sodi"] * (1 - pct)
    return X_new, eligible

# intervention - increase physical activity
def _increase_activity(X: pd.DataFrame, days: int):
    X_new = X.copy()
    eligible = pd.Series(False, index=X_new.index)
    if "activity_days_total" in X_new.columns:
        eligible = X_new["activity_days_total"].notna() & (X_new["activity_days_total"] < 5)
        X_new.loc[eligible, "activity_days_total"] = (X_new.loc[eligible, "activity_days_total"] + days).clip(upper=7)
    if "moderate_days" in X_new.columns:
        X_new.loc[eligible & X_new["moderate_days"].notna(), "moderate_days"] = (X_new.loc[eligible & X_new["moderate_days"].notna(), "moderate_days"] + days).clip(upper=7)
    if "vigorous_days" in X_new.columns:
        X_new.loc[eligible & X_new["vigorous_days"].notna(), "vigorous_days"] = (X_new.loc[eligible & X_new["vigorous_days"].notna(), "vigorous_days"] + days).clip(upper=7)
    return X_new, eligible

# intervention - improve sleep
def _improve_sleep(X: pd.DataFrame, target_hours: float):
    X_new = X.copy()
    eligible_12 = pd.Series(False, index=X_new.index)
    eligible_13 = pd.Series(False, index=X_new.index)
    if "SLD012" in X_new.columns:
        eligible_12 = X_new["SLD012"].notna() & (X_new["SLD012"] < target_hours)
        X_new.loc[eligible_12, "SLD012"] = target_hours
    if "SLD013" in X_new.columns:
        eligible_13 = X_new["SLD013"].notna() & (X_new["SLD013"] < target_hours)
        X_new.loc[eligible_13, "SLD013"] = target_hours
    return X_new, (eligible_12 | eligible_13)

# intervention - reduce alcohol
def _reduce_alcohol(X: pd.DataFrame, pct: float):
    X_new = X.copy()
    eligible = pd.Series(False, index=X_new.index)
    if "diet_alco" in X_new.columns:
        eligible = X_new["diet_alco"].notna() & (X_new["diet_alco"] > 0)
        X_new.loc[eligible, "diet_alco"] = X_new.loc[eligible, "diet_alco"] * (1 - pct)
    return X_new, eligible

# intervention - reduce weight
def _reduce_weight(X: pd.DataFrame, pct: float):
    X_new = X.copy()
    eligible = pd.Series(False, index=X_new.index)
    if "BMXBMI" in X_new.columns:
        eligible = X_new["BMXBMI"].notna() & (X_new["BMXBMI"] >= 25)
        X_new.loc[eligible, "BMXBMI"] = X_new.loc[eligible, "BMXBMI"] * (1 - pct)
    if "BMXWT" in X_new.columns:
        X_new.loc[eligible & X_new["BMXWT"].notna(), "BMXWT"] = X_new.loc[eligible & X_new["BMXWT"].notna(), "BMXWT"] * (1 - pct)
    if "BMXWAIST" in X_new.columns:
        X_new.loc[eligible & X_new["BMXWAIST"].notna(), "BMXWAIST"] = X_new.loc[eligible & X_new["BMXWAIST"].notna(), "BMXWAIST"] * (1 - pct)
    return X_new, eligible

# have about 5 different interventines per relevant category (could add more)
# i wish i couldve been able to automate this but i couldn't acheive that sadly
# results also seem to be a bit off
INTERVENTIONS = {
    "reduce_sodium_5pct": lambda X: _reduce_sodium(X, 0.05),
    "reduce_sodium_10pct": lambda X: _reduce_sodium(X, 0.10),
    "reduce_sodium_15pct": lambda X: _reduce_sodium(X, 0.15),
    "reduce_sodium_20pct": lambda X: _reduce_sodium(X, 0.20),
    "reduce_sodium_25pct": lambda X: _reduce_sodium(X, 0.25),
    "increase_activity_1day": lambda X: _increase_activity(X, 1),
    "increase_activity_2days": lambda X: _increase_activity(X, 2),
    "increase_activity_3days": lambda X: _increase_activity(X, 3),
    "increase_activity_4days": lambda X: _increase_activity(X, 4),
    "increase_activity_5days": lambda X: _increase_activity(X, 5),
    "sleep_at_least_6h": lambda X: _improve_sleep(X, 6),
    "sleep_at_least_6_5h": lambda X: _improve_sleep(X, 6.5),
    "sleep_at_least_7h": lambda X: _improve_sleep(X, 7),
    "sleep_at_least_7_5h": lambda X: _improve_sleep(X, 7.5),
    "sleep_at_least_8h": lambda X: _improve_sleep(X, 8),
    "reduce_alcohol_5pct": lambda X: _reduce_alcohol(X, 0.05),
    "reduce_alcohol_10pct": lambda X: _reduce_alcohol(X, 0.10),
    "reduce_alcohol_15pct": lambda X: _reduce_alcohol(X, 0.15),
    "reduce_alcohol_20pct": lambda X: _reduce_alcohol(X, 0.20),
    "reduce_alcohol_25pct": lambda X: _reduce_alcohol(X, 0.25),
    "reduce_weight_5pct": lambda X: _reduce_weight(X, 0.05),
    "reduce_weight_10pct": lambda X: _reduce_weight(X, 0.10),
    "reduce_weight_15pct": lambda X: _reduce_weight(X, 0.15),
    "reduce_weight_20pct": lambda X: _reduce_weight(X, 0.20),
    "reduce_weight_25pct": lambda X: _reduce_weight(X, 0.25),
}


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def apply_intervention(name: str, X: pd.DataFrame):
    return INTERVENTIONS[name](X)

# runs the recommendation simulation and saves the results and plot with _save
def run_recommendations(features_path: Path, targets_path: Path, baseline_results_path: Path, out_csv: Path, out_png: Path) -> pd.DataFrame:
    features = pd.read_csv(features_path)
    targets = pd.read_csv(targets_path)
    results = pd.read_csv(baseline_results_path)
    best = _best_lifestyle_row(results)

    df = features.merge(targets, on="SEQN", how="inner")
    cols = _get_feature_set_cols(best["feature_set"])
    present = [col for col in cols if col in df.columns]
    X = df[present]
    y = df["hypertension"]
    num_cols, cat_cols = _split_columns(present)

    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=25, stratify=y)
    model = MODEL_FNS[best["model"]](X_train, y_train, num_cols, cat_cols)
    baseline_risk = model.predict_proba(X_test)[:, 1]

    # run the interventions for each profile
    rows = []
    for name, fn in INTERVENTIONS.items():
        X_sim, eligible_mask = fn(X_test)
        if eligible_mask.sum() == 0:
            continue
        simulated_risk = model.predict_proba(X_sim)[:, 1]
        reduction = baseline_risk[eligible_mask] - simulated_risk[eligible_mask]
        rows.append({
            "intervention": name,
            "mean_baseline_risk": baseline_risk[eligible_mask].mean(),
            "mean_post_risk": simulated_risk[eligible_mask].mean(),
            "mean_risk_reduction": reduction.mean(),
            "median_risk_reduction": pd.Series(reduction).median(),
            "pct_people_improved": (reduction > 0).mean(),
            "n_eligible": int(eligible_mask.sum()),
            "model": best["model"],
            "feature_set": best["feature_set"],
            "n_people": len(X_test),
        })

    out = pd.DataFrame(rows).sort_values("mean_risk_reduction", ascending=False)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=out, x="mean_risk_reduction", y="intervention", orient="h")
    plt.title(f"Intervention Ranking: {best['model']} ({best['feature_set']})")
    plt.xlabel("Mean Predicted Risk Reduction")
    plt.ylabel("Intervention")
    _save(out_png)
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    features_path = root / "data" / "processed" / "features.csv"
    targets_path = root / "data" / "processed" / "targets.csv"
    baseline_results_path = root / "reports" / "baseline_results.csv"
    out_csv = root / "reports" / "recommendation_simulation.csv"
    out_png = root / "figures" / "recommendation_ranking.png"
    run_recommendations(features_path, targets_path, baseline_results_path, out_csv, out_png)


if __name__ == "__main__":
    main()
