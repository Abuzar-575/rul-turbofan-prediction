# RUL Prediction for Turbofan Engines with Uncertainty Quantification

## Project overview

We predict the Remaining Useful Life (RUL) of turbofan engines from multivariate sensor time series, and attach an uncertainty estimate to each prediction. The planned models are Random Forest / XGBoost, LSTM / 1D-CNN, MC Dropout and Deep Ensembles. Results are evaluated with RMSE, the NASA asymmetric score, prediction-interval coverage, interval width and calibration plots.

## Dataset

NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set (NASA Prognostics Center of Excellence).

- Repository: https://data.phmsociety.org/nasa/ (item 6, "Turbofan Engine Degradation Simulation Data Set")
- Direct download: https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
- Paper: Saxena et al. (2008), *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*

The data is **not** included in this repository. Download the zip, put it in `data/`, and extract it. We currently work with sub-dataset **FD001** (100 train engines, 100 test engines, one operating condition, one fault mode).

## Phase 2: Data Acquisition & Setup (current status)

The notebook `F26-16.ipynb` runs in Google Colab and does the following:

1. Mounts Google Drive and extracts the dataset zip.
2. Loads `train_FD001`, `test_FD001` and `RUL_FD001` with 26 named columns (unit, cycle, op1-op3, s1-s21).
3. Labels training RUL as (last cycle of the engine) minus (current cycle).
4. Drops constant features: sensors s1, s5, s10, s16, s18, s19 and setting op3.
5. Caps RUL at 125 and applies MinMax scaling (fitted on train only).
6. Builds sliding windows of 30 cycles over 17 features.
7. Saves `X_train_FD001.npy` with shape (17731, 30, 17) and `y_train_FD001.npy` with shape (17731,).
8. Brief EDA: sensor s4 versus cycle, and the RUL label distribution.

## How to run

1. Open `F26-16.ipynb` in Google Colab.
2. Place the dataset zip at `MyDrive/RUL_project/data/` in Google Drive.
3. Run all cells in order.

## Repository contents

```
F26-16.ipynb          Phase 2 notebook
F26-16_Phase2.pdf     Phase 2 report
README.md
.gitignore            excludes data/ and *.npy
```

## Roadmap

- Phase 3 onward: baseline models, LSTM / 1D-CNN, MC Dropout and Deep Ensembles, calibration evaluation.
- Extend the pipeline to FD002-FD004 (per-operating-condition normalisation needed).
