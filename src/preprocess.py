
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple
import numpy as np
import pandas as pd

# apparently the numbers below are downt know codes for NHANES
MISSING_CODES = {7, 9, 77, 99, 777, 999, 7777, 9999}

REQUIRED_MODULES = ["DEMO_J", "BMX_J", "BPX_J", "BPQ_J", "PAQ_J", "ALQ_J", "SLQ_J"]

# usable data
DEMO_COLS = ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "DMDEDUC2", "DMDMARTL"]
BMX_COLS = ["SEQN", "BMXBMI", "BMXWT", "BMXHT", "BMXWAIST"]
BPQ_COLS = ["SEQN", "BPQ020", "BPQ040A", "BPQ050A", "BPQ060", "BPQ070", "BPQ080"]
PAQ_COLS = ["SEQN","PAQ605", "PAQ610", "PAD615","PAQ620", "PAQ625", "PAD630","PAQ635", "PAQ640", "PAD645","PAQ650", "PAQ655", "PAD660","PAQ665", "PAQ670", "PAD675", "PAD680",]
ALQ_COLS = ["SEQN","ALQ111", "ALQ121", "ALQ130", "ALQ142","ALQ151", "ALQ170", "ALQ270", "ALQ280", "ALQ290",]
SLQ_COLS = ["SEQN","SLQ300", "SLQ310", "SLD012","SLQ320", "SLQ330", "SLD013","SLQ030", "SLQ040", "SLQ050", "SLQ120",]

# Nutrient codes shared across both dietary recall day files
DIET_NUTRIENTS = ["KCAL", "PROT", "CARB", "SUGR", "FIBE", "TFAT", "SFAT", "MFAT", "PFAT", "SODI", "ALCO"]

# first, read the xpt (file from the nhanes website)
def _read_xpt(path: Path) -> pd.DataFrame:
    return pd.read_sas(path, format="xport")

# use the codes to replace missing values
def _replace_missing(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].where(~df[col].isin(MISSING_CODES), np.nan)
    return df

# average the multiple BP readings into a single mean systolic and diastolic value per participant
def _mean_bp(df_bpx: pd.DataFrame) -> pd.DataFrame:
    sys_cols = [col for col in df_bpx.columns if col.startswith("BPXSY")]
    dia_cols = [col for col in df_bpx.columns if col.startswith("BPXDI")]
    out = df_bpx[["SEQN"]].copy()
    out["bp_sys_mean"] = df_bpx[sys_cols].mean(axis=1, skipna=True)
    out["bp_dia_mean"] = df_bpx[dia_cols].mean(axis=1, skipna=True)
    return out

# sum up the item-level dietary data into total daily intake per nutrient 
def _sum_diet(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    nutrient_cols = [f"{prefix}{n}" for n in DIET_NUTRIENTS]
    keep = ["SEQN"] + [col for col in nutrient_cols if col in df.columns]
    return df[keep].groupby("SEQN", as_index=False).sum(numeric_only=True)

# average diet intake across the two days where data was taken
def _average_diet(day1: pd.DataFrame, day2: pd.DataFrame) -> pd.DataFrame:
    d1 = day1.set_index("SEQN")
    d2 = day2.set_index("SEQN")
    combined = d1.join(d2, how="outer", lsuffix="_d1", rsuffix="_d2")
    out = pd.DataFrame(index=combined.index)
    for col in d1.columns:
        if col in d2.columns:
            out[f"diet_{col.lower()}"] = combined[[f"{col}_d1", f"{col}_d2"]].mean(axis=1, skipna=True)
        else:
            out[f"diet_{col.lower()}"] = combined[f"{col}_d1"]
    out.reset_index(inplace=True)
    return out

# load everything
def _load_modules(raw_dir: Path) -> dict:
    files = {p.stem.upper(): p for p in raw_dir.glob("*.xpt")}
    missing = [m for m in REQUIRED_MODULES if m not in files]
    if missing:
        raise FileNotFoundError(f"Missing required module(s): {', '.join(missing)}")
    return files

# main preprocessing function (reads xpt, cleans missing codes, writes outputs)
def preprocess(raw_dir: Path, out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / "features.csv"
    targets_path  = out_dir / "targets.csv"

    files = _load_modules(raw_dir)

    demo = _read_xpt(files["DEMO_J"])[DEMO_COLS].copy()
    bmx = _read_xpt(files["BMX_J"])[BMX_COLS].copy()
    bpx = _read_xpt(files["BPX_J"])
    bpq = _read_xpt(files["BPQ_J"])[BPQ_COLS].copy()
    paq = _read_xpt(files["PAQ_J"])[PAQ_COLS].copy()
    alq = _read_xpt(files["ALQ_J"])[ALQ_COLS].copy()
    slq = _read_xpt(files["SLQ_J"])[SLQ_COLS].copy()

    # the optonal diet files
    dr1_sum = None
    dr2_sum = None
    if "DR1IFF_J" in files:
        dr1_sum = _sum_diet(_read_xpt(files["DR1IFF_J"]), "DR1I")
    if "DR2IFF_J" in files:
        dr2_sum = _sum_diet(_read_xpt(files["DR2IFF_J"]), "DR2I")

    diet = None
    if dr1_sum is not None and dr2_sum is not None:
        dr1_norm = dr1_sum.rename(columns=lambda c: c.replace("DR1I", "NUTR_"))
        dr2_norm = dr2_sum.rename(columns=lambda c: c.replace("DR2I", "NUTR_"))
        diet = _average_diet(
            dr1_norm.rename(columns=lambda c: c.replace("NUTR_", "")),
            dr2_norm.rename(columns=lambda c: c.replace("NUTR_", "")),
        )
    elif dr1_sum is not None:
        diet = dr1_sum
    elif dr2_sum is not None:
        diet = dr2_sum.rename(columns={c: c.replace("DR2I", "DR1I") for c in dr2_sum.columns})

    demo = _replace_missing(demo, ["RIAGENDR", "RIDRETH1", "DMDEDUC2", "DMDMARTL"])
    bpq = _replace_missing(bpq, bpq.columns)
    paq = _replace_missing(paq, paq.columns)
    alq = _replace_missing(alq, alq.columns)
    slq = _replace_missing(slq, slq.columns)

    # the sleep quality variables have too many missing values to be remotely useful
    slq = slq.drop(columns=["SLQ300", "SLQ310", "SLQ320", "SLQ330"])

    paq["vigorous_days"] = paq[["PAD615", "PAD645"]].sum(axis=1, skipna=True)
    paq["moderate_days"] = paq[["PAD630", "PAD660"]].sum(axis=1, skipna=True)
    paq["activity_days_total"] = paq[["vigorous_days", "moderate_days"]].sum(axis=1, skipna=True)

    # finally merge everything
    features = demo.merge(bmx,  on="SEQN", how="left")
    features = features.merge(bpq, on="SEQN", how="left")
    features = features.merge(paq, on="SEQN", how="left")
    features = features.merge(alq, on="SEQN", how="left")
    features = features.merge(slq, on="SEQN", how="left")
    features = features.merge(_mean_bp(bpx), on="SEQN", how="left")
    if diet is not None:
        features = features.merge(diet, on="SEQN", how="left")

    # follows AHA 2017 guidelines: stage-1 hypertension threshold is 130/80.
    hyp_bp = (features["bp_sys_mean"] >= 130) | (features["bp_dia_mean"] >= 80)
    hyp_self = features["BPQ020"] == 1
    hyp_meds = features["BPQ040A"] == 1
    features["hypertension"] = (hyp_bp | hyp_self | hyp_meds).astype(float)

    targets = features[["SEQN", "hypertension"]].copy()
    features = features.drop(columns=["hypertension"])

    # save raw features before dropping leakage columns (for raw EDA)
    features.to_csv(out_dir / "features_raw.csv", index=False)

    # drop columns that would leak label information at inference time -- SUPER IMPORTANT*****
    BPQ_LEAKAGE_COLS = ["BPQ020", "BPQ040A", "BPQ050A", "BPQ060", "BPQ070", "BPQ080"]
    BP_MEASUREMENT_COLS = ["bp_sys_mean", "bp_dia_mean"]
    drop_cols = [col for col in BPQ_LEAKAGE_COLS + BP_MEASUREMENT_COLS if col in features.columns]
    features = features.drop(columns=drop_cols)
    features.to_csv(features_path, index=False)
    targets.to_csv(targets_path,  index=False)
    return features_path, targets_path

def main() -> None:
    from .config import DATA_PROCESSED, DATA_RAW
    preprocess(DATA_RAW, DATA_PROCESSED)


if __name__ == "__main__":
    main()
