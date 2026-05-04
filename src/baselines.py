
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from .models import train_logistic_regression, train_random_forest, train_xgboost


DEMOGRAPHIC_COLS = ["RIDAGEYR", "RIAGENDR", "RIDRETH1", "DMDEDUC2", "DMDMARTL",]

# lifestyle is everything else: body measurements, physical activity,
# alcohol use, sleep, and dietary intake as welll
LIFESTYLE_COLS = [
    # anthropometrics
    "BMXBMI", "BMXWT", "BMXHT", "BMXWAIST",
    # physical activity (raw + derived)
    "PAQ605", "PAQ610", "PAD615", "PAQ620", "PAQ625", "PAD630", "PAQ635", "PAQ640", "PAD645",
    "PAQ650", "PAQ655", "PAD660", "PAQ665", "PAQ670", "PAD675", "PAD680","activity_days_total",
    # alcohol
    "ALQ111", "ALQ121", "ALQ130", "ALQ142", "ALQ151", "ALQ170", "ALQ270", "ALQ280", "ALQ290",
    # sleep
    "SLD012", "SLD013", "SLQ030", "SLQ040", "SLQ050", "SLQ120",
    # diet
    "diet_kcal", "diet_prot", "diet_carb", "diet_sugr", "diet_fibe", "diet_tfat", "diet_sfat", "diet_mfat", "diet_pfat", "diet_sodi", "diet_alco",
]


def _evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

# separate numerical and categorical
def _split_columns(cols: List[str]) -> Tuple[List[str], List[str]]:
    CAT_COLS = {"RIAGENDR", "RIDRETH1", "DMDEDUC2", "DMDMARTL"}
    cat = [col for col in cols if col in CAT_COLS]
    num = [col for col in cols if col not in CAT_COLS]
    return num, cat

# runs the baseline model(s)
# should save results too
def run_baselines(features_path: Path, targets_path: Path, out_csv: Path) -> pd.DataFrame:
    features = pd.read_csv(features_path)
    targets = pd.read_csv(targets_path)
    df = features.merge(targets, on="SEQN", how="inner")

    y = df["hypertension"]
    results = []
    model_fns = [("logistic_regression", train_logistic_regression), ("random_forest", train_random_forest), ("xgboost", train_xgboost)]

    feature_sets = [("demographic_only", DEMOGRAPHIC_COLS), ("lifestyle_only", LIFESTYLE_COLS), ("combined", DEMOGRAPHIC_COLS + LIFESTYLE_COLS),]

    for name, cols in feature_sets:
        present = [col for col in cols if col in df.columns]
        X = df[present]
        num_cols, cat_cols = _split_columns(present)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25, stratify=y)

        for model_name, model_fn in model_fns:
            model = model_fn(X_train, y_train, num_cols, cat_cols)
            metrics = _evaluate(model, X_test, y_test)
            metrics.update({"model": model_name, "feature_set": name, "n_features": len(present)})
            results.append(metrics)
            print(f"[{model_name}][{name}] AUC={metrics['roc_auc']:.3f}  F1={metrics['f1']:.3f}  Acc={metrics['accuracy']:.3f}")

    results_df = pd.DataFrame(results)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_csv, index=False)
    return results_df


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    features_path = root / "data" / "processed" / "features.csv"
    targets_path = root / "data" / "processed" / "targets.csv"
    out_csv = root / "reports" / "baseline_results.csv"
    run_baselines(features_path, targets_path, out_csv)


if __name__ == "__main__":
    main()