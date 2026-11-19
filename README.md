# ✈️ Turbofan Engine RUL Prediction & Uncertainty Quantification (Project F26-16)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow / Keras](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-NASA%20C--MAPSS%20FD001-00D4FF)](#dataset--leakage-free-preprocessing)

**Deep Sequence Prognostics and Risk-Aware Predictive Maintenance on NASA C-MAPSS `FD001`**

An end-to-end machine learning and deep learning prognostics system that estimates **Remaining Useful Life (RUL)** for commercial turbofan engines alongside calibrated **95% epistemic confidence intervals** via **Monte Carlo (MC) Dropout**. Instead of relying on overconfident point predictions, the system triggers maintenance decisions using the conservative lower confidence bound ($\text{CI}_{95,\text{lower}}$), deployed through an interactive **AeroGuard RUL** Streamlit analytics dashboard.


## 🔬 Key Technical Contributions
1. **Strict Engine-Level Leakage Prevention:** Splits the 100 `FD001` training engines into an 80/20 engine-disjoint train/validation split prior to fitting `MinMaxScaler` or generating sliding sequences, eliminating window-overlap data leakage.
2. **Domain-Informed Feature Selection & Piecewise Linear RUL Target:** Removes 7 invariant/dead telemetry channels to retain 17 active sensors/settings (`window_size = 30` cycles) and applies a piecewise linear degradation ceiling at `RUL_CAP = 125` cycles.
3. **Point Accuracy vs. Uncertainty Calibration Benchmark:** Evaluates tabular feature-engineered baselines (**Random Forest**, **Gradient Boosting**) against deep probabilistic models (**Stacked LSTM + MC Dropout**, **Deep Ensemble**), demonstrating why pure MSE ensembles suffer from severe under-dispersion (`27.0%` PICP) while MC Dropout achieves well-calibrated `85.0%` coverage.
4. **Risk-Aware Maintenance Decision Engine:** Automates maintenance scheduling (`NORMAL OPERATION`, `INSPECT & MONITOR CLOSELY`, `SCHEDULE IMMEDIATE MAINTENANCE`) by comparing the lower 95% confidence bound against a configurable safety threshold.

---

## 📊 Experimental Benchmark Results (`FD001` Test Set — 100 Engines)

| Model Architecture | RMSE (Cycles) ↓ | MAE (Cycles) ↓ | NASA Asymmetric Score ↓ | 95% PICP (%) ↑ | 95% MPIW (Cycles) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** *(Engineered Window Features)* | 13.33 | 9.43 | 308.98 | — | — |
| **Gradient Boosting** *(Engineered Window Features)* | **12.42** | **9.30** | **244.50** | — | — |
| **Deep Ensemble** *(5 × Stacked LSTM)* | 12.99 | 9.56 | 309.55 | 27.0% | 7.76 |
| **Stacked LSTM + MC Dropout** *(Deployed, $T=100$)* | 13.79 | 9.95 | 412.86 | **85.0%** | **39.77** |

### Case Study: Test Engine `#34` (Near-Failure Validation)
* **Ground-Truth RUL:** `7` cycles
* **MC Dropout Mean Prediction:** `4.81` cycles ($\pm 2.29$ cycles std)
* **95% Confidence Interval:** `[0.31, 9.30]` cycles *(cleanly brackets true RUL)*
* **Automated Decision:** `SCHEDULE IMMEDIATE MAINTENANCE` ($\text{CI}_{95,\text{lower}} = 0.31 \le 20$ cycles)

---

## 📁 Repository Structure

```text
rul-turbofan-prediction/
├── .streamlit/
│   └── config.toml                          # Midnight Aerospace UI theme configuration
├── models/
│   ├── lstm_mc_dropout_FD001.keras          # Trained Stacked LSTM (64 → 32) with Dropout(0.2)
│   ├── scaler_FD001.pkl                     # Fitted MinMaxScaler [0, 1] on 17 active features
│   └── features_FD001.json                  # Window size, RUL cap, and feature schema metadata
├── Phase 1/                                 # Phase 1 project proposal & problem formulation
├── Phase 2/                                 # Phase 2 exploratory data analysis & preprocessing docs
├── Phase 3/                                 # Phase 3 baseline modeling & architecture design docs
├── Phase 4/                                 # Phase 4 uncertainty quantification & evaluation docs
├── Phase 5/
│   └── F26_16_Final_Technical_Report.pdf    # Final technical report (results, error analysis & theory)
├── app.py                                   # AeroGuard RUL interactive Streamlit web dashboard
├── train_rul_uq.py                          # End-to-end training, benchmarking & UQ evaluation script
├── requirements.txt                         # Python package dependencies
├── .gitignore                               # Excludes local venv, caches, and raw zip archives
└── README.md                                # Project documentation