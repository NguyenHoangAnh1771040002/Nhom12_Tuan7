import os
import time
import papermill as pm
import gc

# Run notebooks end-to-end (classification + regression + ARIMA)
os.makedirs("notebooks/runs", exist_ok=True)

# Sử dụng kernel hiện tại (python3) hoặc kernel của môi trường đang active
KERNEL = "python3"

def run_notebook(input_path, output_path, params):
    """Execute notebook with proper cleanup."""
    print(f"\n{'='*60}")
    print(f"Running: {input_path}")
    print(f"{'='*60}")
    pm.execute_notebook(
        input_path,
        output_path,
        parameters=params,
        language="python",
        kernel_name=KERNEL,
    )
    # Force garbage collection and small delay for kernel cleanup
    gc.collect()
    time.sleep(2)
    print(f"[OK] Completed: {output_path}")

# 1. Preprocessing and EDA
run_notebook(
    "notebooks/preprocessing_and_eda.ipynb",
    "notebooks/runs/preprocessing_and_eda_run.ipynb",
    dict(
        USE_UCIMLREPO=False,
        RAW_ZIP_PATH="data/raw/PRSA2017_Data_20130301-20170228.zip",
        OUTPUT_CLEANED_PATH="data/processed/cleaned.parquet",
        LAG_HOURS=[1, 3, 24],
    ),
)

# 2. Feature Preparation
run_notebook(
    "notebooks/feature_preparation.ipynb",
    "notebooks/runs/feature_preparation_run.ipynb",
    dict(
        CLEANED_PATH="data/processed/cleaned.parquet",
        OUTPUT_DATASET_PATH="data/processed/dataset_for_clf.parquet",
        DROP_ROWS_WITHOUT_TARGET=True,
    ),
)

# 3. Classification Modelling
run_notebook(
    "notebooks/classification_modelling.ipynb",
    "notebooks/runs/classification_modelling_run.ipynb",
    dict(
        DATASET_PATH="data/processed/dataset_for_clf.parquet",
        CUTOFF="2017-01-01",
        METRICS_PATH="data/processed/metrics.json",
        PRED_SAMPLE_PATH="data/processed/predictions_sample.csv",
    ),
)

# 4. Regression Modelling (supervised, lag-based)
run_notebook(
    "notebooks/regression_modelling.ipynb",
    "notebooks/runs/regression_modelling_run.ipynb",
    dict(
        USE_UCIMLREPO=False,
        RAW_ZIP_PATH="data/raw/PRSA2017_Data_20130301-20170228.zip",
        LAG_HOURS=[1, 3, 24],
        HORIZON=1,
        TARGET_COL="PM2.5",
        OUTPUT_REG_DATASET_PATH="data/processed/dataset_for_regression.parquet",
        CUTOFF="2017-01-01",
        MODEL_OUT="regressor.joblib",
        METRICS_OUT="regression_metrics.json",
        PRED_SAMPLE_OUT="regression_predictions_sample.csv",
    ),
)

# 5. Time-series forecasting with ARIMA
run_notebook(
    "notebooks/arima_forecasting.ipynb",
    "notebooks/runs/arima_forecasting_run.ipynb",
    dict(
        RAW_ZIP_PATH="data/raw/PRSA2017_Data_20130301-20170228.zip",
        STATION="Aotizhongxin",
        VALUE_COL="PM2.5",
        CUTOFF="2017-01-01",
        P_MAX=3,
        Q_MAX=3,
        D_MAX=2,
        IC="aic",
        ARTIFACTS_PREFIX="arima_pm25",
    ),
)

# 6. Comparison: Regression vs ARIMA (Chu de phat trien 1)
run_notebook(
    "notebooks/comparison_regression_vs_arima.ipynb",
    "notebooks/runs/comparison_regression_vs_arima_run.ipynb",
    dict(),  # No parameters needed - uses saved predictions
)

print("\n" + "="*60)
print("[OK] Da chay xong pipeline (classification + regression + ARIMA + comparison)")
print("="*60)
