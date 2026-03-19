# ECE 5424 Capstone - Hypertension Prediction from Lifestyle Data

Author: Muhammad Athar

Goal: Build machine learning model(s) that predict incident hypertension risk from lifestyle behaviors and compare against unmodifiable demographic baselines.

## Directory Structure
- `data/raw`: raw NHANES data files
- `data/processed`: feature-engineered datasets
- `reports`: csv files for overall results, including baseline
- `figures`: charts/plots for EDA and results
- `src`: source code

## Main Setup
- Need Python 3.9+

## Run
- Build processed datasets (also writes `features_raw.csv` for raw EDA):
  - `python -m src.preprocess`
- Generate EDA figures (model-safe features):
  - `python -m src.eda`

## Optional: Raw EDA (for BPQ and BP included within the feature set)
If you want EDA that includes BPQ and BP measurement columns, run:
- `python -c "from src.eda import run_eda_raw; from src.config import DATA_PROCESSED, FIGURES, PROJECT_ROOT; run_eda_raw(DATA_PROCESSED, FIGURES / 'raw', PROJECT_ROOT / 'reports' / 'raw')"`

## Run baseline model(s):
- `python -m src.baselines`

## Data
Dataset: NHANES 2017-2018

## Optional, but Nice to See
To view dataset, install 'Clinical Data Viewer for SAS Family' on VSCode (My Preferred IDE)