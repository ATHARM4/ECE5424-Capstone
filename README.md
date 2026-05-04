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

## Install Dependencies
  - `pip install -r requirements.txt`

## Reproducibility Notes
- This repository does not store serialized model checkpoints by default.
- All models and result artifacts can be regenerated locally by running the scripts below in order.

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

## Run model interpretation
  - `python -m src.interpretability`

## Run intervention simulation / recommendations
  - `python -m src.recommendations`

## Run final output figures
  - `python -m src.final_outputs`

## Optional: Decision Curve Figure
  - `python -m src.decision_curve`

## Example Notebook
- A simple summary notebook is included at `notebooks/project_summary.ipynb`
- In VS Code, open the notebook and run all cells with the project Python environment selected
- In Jupyter, from the repo root run:
  - `python -m notebook`
- The notebook reads existing CSV outputs from `reports/`, so run the scripts above first if those files are not already present

## Suggested Reproduction Order
1. `python -m src.preprocess`
2. `python -m src.eda`
3. `python -m src.baselines`
4. `python -m src.interpretability`
5. `python -m src.recommendations`
6. `python -m src.final_outputs`
7. `python -m src.decision_curve` (optional)

## Data
Dataset: NHANES 2017-2018

## Optional, but Nice to See
To view dataset, install 'Clinical Data Viewer for SAS Family' on VSCode (My Preferred IDE)