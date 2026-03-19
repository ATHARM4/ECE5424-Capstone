
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

# generates all the outputs (maps, plots, charts, etc.) for data analysis
def _run_eda_core(df: pd.DataFrame, figures_dir: Path, reports_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")

    # outcome "prevalence" (really just counts since it's a binary label)
    counts = df["hypertension"].value_counts(dropna=False)
    counts.to_csv(reports_dir / "outcome_prevalence.csv")

    plt.figure(figsize=(5, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values)
    plt.title("Hypertension Outcome Prevalence")
    plt.xlabel("Hypertension (0 = No, 1 = Yes)")
    plt.ylabel("Count")
    _save(figures_dir / "outcome_prevalence.png")

    # missingness
    missing = df.isna().mean().sort_values(ascending=False)
    missing.to_csv(reports_dir / "missingness_rate.csv")

    top20 = missing.head(20)
    plt.figure(figsize=(8, 6))
    sns.barplot(y=top20.index, x=top20.values, orient="h")
    plt.title("Top 20 Features by Missingness Rate")
    plt.xlabel("Fraction Missing")
    plt.ylabel("Feature")
    _save(figures_dir / "missingness_top20.png")

    # variable distributions
    to_plot = [
        ("BMXBMI", "bmi_distribution.png", "BMI Distribution"), ("RIDAGEYR", "age_distribution.png", "Age Distribution"),
        ("diet_sodi", "sodium_distribution.png", "Dietary Sodium (daily avg)"), ("SLD012", "sleep_distribution.png", "Weeknight Sleep Duration (hrs)"),
    ]
    for col, fname, title in to_plot:
        if col not in df.columns:
            continue
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), bins=30, kde=True)
        plt.title(title)
        plt.xlabel(col)
        plt.ylabel("Count")
        _save(figures_dir / fname)

    # create correlation heatmap(s)
    numeric_cols = [
        "RIDAGEYR", "BMXBMI", "BMXWAIST", "bp_sys_mean", "bp_dia_mean",
        "diet_kcal", "diet_sodi", "diet_sugr", "SLD012", "activity_days_total",
    ]
    present = [col for col in numeric_cols if col in df.columns]
    if len(present) >= 3:
        corr = df[present].corr(numeric_only=True)
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap (Selected Numeric Features)")
        _save(figures_dir / "correlation_heatmap.png")

# actually run the eda
def run_eda(processed_dir: Path, figures_dir: Path, reports_dir: Path) -> None:
    features = pd.read_csv(processed_dir / "features.csv")
    targets = pd.read_csv(processed_dir / "targets.csv")
    df = features.merge(targets, on="SEQN", how="inner")
    _run_eda_core(df, figures_dir, reports_dir)

# separated out so we can run it on the raw features with the bp_sys/dia means still present (essentially a second version)
def run_eda_raw(processed_dir: Path, figures_dir: Path, reports_dir: Path) -> None:
    raw_path = processed_dir / "features_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            "features_raw.csv not found"
        )

    features = pd.read_csv(raw_path)
    targets = pd.read_csv(processed_dir / "targets.csv")
    df = features.merge(targets, on="SEQN", how="inner")
    _run_eda_core(df, figures_dir, reports_dir)


def main() -> None:
    from .config import DATA_PROCESSED, FIGURES, PROJECT_ROOT
    run_eda(DATA_PROCESSED, FIGURES, PROJECT_ROOT / "reports")


if __name__ == "__main__":
    main()