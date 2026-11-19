import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

# ---------------------------------------------------------
# 1. Page Configuration & Cohesive Midnight Aerospace Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Turbofan Engine RUL Prediction & Uncertainty Quantification",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Force cohesive background across main canvas, header, and sidebar */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #080E1A;
        color: #F1F5F9;
    }
    [data-testid="stHeader"] {
        background-color: rgba(8, 14, 26, 0.85);
        backdrop-filter: blur(8px);
    }
    [data-testid="stSidebar"] {
        background-color: #0D1525;
        border-right: 1px solid #1E293B;
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.2rem;
    }

    /* Project Title Hero Banner */
    .project-hero {
        background: linear-gradient(120deg, #0F172A 0%, #132238 55%, #0B3558 100%);
        border: 1px solid rgba(0, 212, 255, 0.28);
        border-radius: 16px;
        padding: 1.65rem 2rem;
        margin-bottom: 1.35rem;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35);
    }
    .project-eyebrow {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #00D4FF;
        margin-bottom: 0.35rem;
    }
    .project-main-title {
        font-size: 1.95rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    .project-sub {
        font-size: 0.95rem;
        color: #CBD5E1;
        margin-top: 0.45rem;
        margin-bottom: 0.85rem;
    }
    .meta-chip {
        display: inline-block;
        background: rgba(8, 14, 26, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.25);
        color: #E2E8F0;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .meta-chip span {
        color: #00D4FF;
    }

    /* Elevated KPI Cards matching #111927 surface */
    .kpi-card {
        background: #111927;
        border: 1px solid #1F2D42;
        border-top: 3px solid #00D4FF;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
    }
    .kpi-label {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.15;
    }
    .kpi-unit {
        font-size: 0.92rem;
        font-weight: 500;
        color: #94A3B8;
    }
    .kpi-sub {
        font-size: 0.79rem;
        color: #64748B;
        margin-top: 0.3rem;
    }

    /* Risk Status Banners */
    .status-banner {
        padding: 1rem 1.3rem;
        border-radius: 12px;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.95rem;
    }
    .risk-high {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.18) 0%, rgba(17, 25, 39, 0.95) 100%);
        border: 1px solid rgba(239, 68, 68, 0.45);
        border-left: 5px solid #EF4444;
        color: #FEE2E2;
    }
    .risk-mod {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.18) 0%, rgba(17, 25, 39, 0.95) 100%);
        border: 1px solid rgba(245, 158, 11, 0.45);
        border-left: 5px solid #F59E0B;
        color: #FEF3C7;
    }
    .risk-low {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.18) 0%, rgba(17, 25, 39, 0.95) 100%);
        border: 1px solid rgba(16, 185, 129, 0.45);
        border-left: 5px solid #10B981;
        color: #D1FAE5;
    }
    .risk-pill {
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.05em;
    }

    /* Style Preset Buttons & Tabs cleanly */
    .stButton > button {
        background-color: #111927;
        color: #F1F5F9;
        border: 1px solid #26354D;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #00D4FF;
        color: #00D4FF;
        background-color: #162236;
    }
    button[data-baseweb="tab"] {
        font-weight: 600;
        color: #94A3B8;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00D4FF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_DIR = "models"
INDEX_COLS = ["unit_id", "time_cycles"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
ALL_CMAPSS_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

SENSOR_DESCRIPTIONS = {
    "setting_1": "Operational Setting 1 (Altitude / Mach)",
    "setting_2": "Operational Setting 2 (Throttle / Speed)",
    "s_2": "s_2 — LPC Outlet Temp (T24)",
    "s_3": "s_3 — HPC Outlet Temp (T30)",
    "s_4": "s_4 — LPT Outlet Temp / EGT (T50)",
    "s_7": "s_7 — HPC Outlet Total Pressure (P30)",
    "s_8": "s_8 — Physical Fan Speed (Nf)",
    "s_9": "s_9 — Physical Core Speed (Nc)",
    "s_11": "s_11 — HPC Static Pressure (Ps30)",
    "s_12": "s_12 — Fuel Flow / Ps30 Ratio (phi)",
    "s_13": "s_13 — Corrected Fan Speed (NRf)",
    "s_14": "s_14 — Corrected Core Speed (NRc)",
    "s_15": "s_15 — Bypass Ratio (BPR)",
    "s_17": "s_17 — Bleed Enthalpy (htBleed)",
    "s_20": "s_20 — HPT Coolant Bleed (W31)",
    "s_21": "s_21 — LPT Coolant Bleed (W32)",
}


@st.cache_resource
def load_artifacts():
    with open(f"{MODEL_DIR}/features_FD001.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    scaler = joblib.load(f"{MODEL_DIR}/scaler_FD001.pkl")
    model = tf.keras.models.load_model(f"{MODEL_DIR}/lstm_mc_dropout_FD001.keras")
    return model, scaler, meta


try:
    model, scaler, meta = load_artifacts()
except Exception as e:
    st.error(
        f"Failed to load model artifacts from `{MODEL_DIR}/`. "
        f"Make sure `lstm_mc_dropout_FD001.keras`, `scaler_FD001.pkl`, and `features_FD001.json` are inside `models/`.\n\nError: {e}"
    )
    st.stop()

WINDOW_SIZE = meta["window_size"]
RUL_CAP = meta["rul_cap"]
SELECTED_FEATURES = meta["selected_features"]


# ---------------------------------------------------------
# 2. Core Inference & Data Helpers (100% Unchanged Logic)
# ---------------------------------------------------------
def run_mc_dropout_inference(engine_df: pd.DataFrame, n_iter: int = 100, safety_threshold: int = 20):
    scaled_vals = scaler.transform(engine_df[SELECTED_FEATURES])

    if len(scaled_vals) < WINDOW_SIZE:
        pad_len = WINDOW_SIZE - len(scaled_vals)
        pad = np.repeat(scaled_vals[0:1], pad_len, axis=0)
        window = np.vstack([pad, scaled_vals])
    else:
        window = scaled_vals[-WINDOW_SIZE:]

    X_batch = np.repeat(np.expand_dims(window, axis=0), n_iter, axis=0).astype(np.float32)
    preds = model(X_batch, training=True).numpy().flatten()

    mean_rul = float(np.clip(np.mean(preds), 0, RUL_CAP))
    std_rul = float(np.std(preds))
    lower_95 = float(max(0.0, mean_rul - 1.96 * std_rul))
    upper_95 = float(min(RUL_CAP, mean_rul + 1.96 * std_rul))

    if lower_95 <= safety_threshold:
        status = "SCHEDULE IMMEDIATE MAINTENANCE"
        risk_level = "HIGH RISK"
    elif mean_rul <= safety_threshold * 2:
        status = "INSPECT & MONITOR CLOSELY"
        risk_level = "MODERATE RISK"
    else:
        status = "NORMAL OPERATION"
        risk_level = "LOW RISK"

    return {
        "predicted_rul": round(mean_rul, 2),
        "uncertainty_std": round(std_rul, 2),
        "ci_95_lower": round(lower_95, 2),
        "ci_95_upper": round(upper_95, 2),
        "status": status,
        "risk_level": risk_level,
        "mc_samples": preds,
    }


def generate_demo_engine_df(degradation_stage: float = 0.80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    cycles = 60
    t = np.linspace(0, degradation_stage, cycles)

    norm_traj = np.zeros((cycles, len(SELECTED_FEATURES)))
    for j in range(len(SELECTED_FEATURES)):
        direction = 1.0 if j % 2 == 0 else -1.0
        base = 0.30 if direction > 0 else 0.70
        norm_traj[:, j] = np.clip(base + direction * 0.50 * (t ** 1.4) + rng.normal(0, 0.015, cycles), 0.02, 0.98)

    raw_vals = scaler.inverse_transform(norm_traj)
    df = pd.DataFrame(raw_vals, columns=SELECTED_FEATURES)
    df.insert(0, "time_cycles", np.arange(1, cycles + 1))
    df.insert(0, "unit_id", 1)
    return df


def parse_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)

    if df.shape[1] == 26 and "unit_id" not in df.columns:
        df.columns = ALL_CMAPSS_COLS

    missing = [col for col in SELECTED_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Uploaded file is missing required columns: {missing}")

    if "unit_id" not in df.columns:
        df["unit_id"] = 1
    if "time_cycles" not in df.columns:
        df["time_cycles"] = np.arange(1, len(df) + 1)

    return df


# ---------------------------------------------------------
# 3. Sidebar Controls & Model Summary
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### Telemetry Input Mode")
    data_mode = st.radio(
        "Select Data Source",
        ["Interactive Demo Simulator", "Upload C-MAPSS File (.txt / .csv)"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Inference & Risk Parameters")
    n_mc_passes = st.slider(
        "MC Dropout Forward Passes (T)",
        min_value=20,
        max_value=200,
        value=100,
        step=10,
        help="Number of stochastic forward passes with dropout active during inference.",
    )
    safety_threshold = st.slider(
        "Safety Threshold (Lower 95% CI Cycles)",
        min_value=5,
        max_value=50,
        value=20,
        step=1,
        help="Immediate maintenance triggers when the lower 95% confidence bound drops to or below this limit.",
    )

    st.markdown("---")
    st.markdown("### Model & Dataset Spec")
    st.markdown(
        f"""
        <div style="background:#111927;border:1px solid #1F2D42;border-radius:10px;padding:0.9rem 1rem;font-size:0.85rem;color:#CBD5E1;line-height:1.65;">
            <div><strong style="color:#00D4FF;">Dataset:</strong> NASA C-MAPSS {meta['dataset']}</div>
            <div><strong style="color:#00D4FF;">Backbone:</strong> Stacked LSTM (64 → 32)</div>
            <div><strong style="color:#00D4FF;">UQ Engine:</strong> MC Dropout (p = 0.20)</div>
            <div><strong style="color:#00D4FF;">Window / Cap:</strong> {WINDOW_SIZE} cycles / {RUL_CAP} cycles</div>
            <div><strong style="color:#00D4FF;">Active Sensors:</strong> {len(SELECTED_FEATURES)} features</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# 4. Main Project Title Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="project-hero">
        <div class="project-eyebrow">Predictive Maintenance & Prognostics • Project F26-16</div>
        <h1 class="project-main-title">✈️ Turbofan Engine RUL Prediction & Uncertainty Quantification</h1>
        <p class="project-sub">
            Deep Sequence Prognostics on NASA C-MAPSS FD001 with Calibrated Epistemic Confidence Intervals & Risk-Aware Maintenance Scheduling
        </p>
        <div>
            <span class="meta-chip">Model: <span>LSTM + MC Dropout</span></span>
            <span class="meta-chip">Calibration: <span>95% Confidence Interval</span></span>
            <span class="meta-chip">Decision Rule: <span>Lower-Bound Safety Trigger</span></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 5. Telemetry Selection Bar (Simulator Presets or File Upload)
# ---------------------------------------------------------
engine_df = None
full_df = None
selected_unit = 1

if data_mode == "Upload C-MAPSS File (.txt / .csv)":
    up_col1, up_col2 = st.columns([2.2, 1])
    with up_col1:
        uploaded = st.file_uploader(
            "Upload `test_FD001.txt` or a custom engine telemetry `.csv` file",
            type=["txt", "csv"],
        )
    if uploaded is not None:
        try:
            full_df = parse_uploaded_file(uploaded)
            unit_ids = sorted(full_df["unit_id"].unique().tolist())
            with up_col2:
                selected_unit = st.selectbox(
                    "Select Engine Unit ID",
                    unit_ids,
                    index=min(33, len(unit_ids) - 1),
                    help="Defaults to Engine #34 when loading test_FD001.txt.",
                )
            engine_df = full_df[full_df["unit_id"] == selected_unit].sort_values("time_cycles").reset_index(drop=True)
        except Exception as err:
            st.error(f"File parsing error: {err}")
            st.stop()
    else:
        st.info("Upload `test_FD001.txt` above, or switch to **Interactive Demo Simulator** in the left sidebar to explore presets.")
        st.stop()
else:
    if "demo_stage" not in st.session_state:
        st.session_state.demo_stage = 0.80

    st.markdown(
        "<div style='font-size:0.84rem;font-weight:600;color:#94A3B8;margin-bottom:0.4rem;'>SIMULATED ENGINE WEAR PROFILE (QUICK PRESETS OR FINE SLIDER)</div>",
        unsafe_allow_html=True,
    )
    p_col1, p_col2, p_col3, p_col4 = st.columns([1, 1, 1, 2.3])
    with p_col1:
        if st.button("🟢 Healthy Cruise (0.20)", width="stretch"):
            st.session_state.demo_stage = 0.20
    with p_col2:
        if st.button("🟡 Mid-Life Wear (0.55)", width="stretch"):
            st.session_state.demo_stage = 0.55
    with p_col3:
        if st.button("🔴 Near Failure (0.85)", width="stretch"):
            st.session_state.demo_stage = 0.85
    with p_col4:
        stage = st.slider(
            "Simulated Degradation Stage",
            min_value=0.10,
            max_value=0.95,
            value=st.session_state.demo_stage,
            step=0.05,
            label_visibility="collapsed",
        )
        st.session_state.demo_stage = stage

    engine_df = generate_demo_engine_df(degradation_stage=st.session_state.demo_stage)


# ---------------------------------------------------------
# 6. Run Inference & Render Risk Banner + Matched KPI Cards
# ---------------------------------------------------------
result = run_mc_dropout_inference(engine_df, n_iter=n_mc_passes, safety_threshold=safety_threshold)

if result["risk_level"] == "HIGH RISK":
    st.markdown(
        f"""
        <div class="status-banner risk-high">
            <div><strong>🚨 {result['status']}</strong> — Lower 95% bound (<b>{result['ci_95_lower']} cycles</b>) is at or below the <b>{safety_threshold}-cycle</b> safety threshold.</div>
            <span class="risk-pill" style="background:rgba(239,68,68,0.28);color:#FCA5A5;">{result['risk_level']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif result["risk_level"] == "MODERATE RISK":
    st.markdown(
        f"""
        <div class="status-banner risk-mod">
            <div><strong>⚠️ {result['status']}</strong> — Mean RUL (<b>{result['predicted_rul']} cycles</b>) is inside the 2× caution buffer ({safety_threshold * 2} cycles).</div>
            <span class="risk-pill" style="background:rgba(245,158,11,0.28);color:#FCD34D;">{result['risk_level']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="status-banner risk-low">
            <div><strong>✅ {result['status']}</strong> — Predicted RUL and lower 95% confidence bound are safely above maintenance limits.</div>
            <span class="risk-pill" style="background:rgba(16,185,129,0.28);color:#6EE7B7;">{result['risk_level']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

interval_width = round(result["ci_95_upper"] - result["ci_95_lower"], 2)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Predicted Mean RUL</div>
            <div class="kpi-value">{result['predicted_rul']} <span class="kpi-unit">cycles</span></div>
            <div class="kpi-sub">Posterior mean across T={n_mc_passes} passes</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""
        <div class="kpi-card" style="border-top-color:#A855F7;">
            <div class="kpi-label">Epistemic Uncertainty (±1σ)</div>
            <div class="kpi-value">± {result['uncertainty_std']} <span class="kpi-unit">cycles</span></div>
            <div class="kpi-sub">Predictive standard deviation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""
        <div class="kpi-card" style="border-top-color:#10B981;">
            <div class="kpi-label">95% Confidence Interval</div>
            <div class="kpi-value">[{result['ci_95_lower']}, {result['ci_95_upper']}]</div>
            <div class="kpi-sub">Interval width (MPIW): {interval_width} cycles</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""
        <div class="kpi-card" style="border-top-color:#F59E0B;">
            <div class="kpi-label">Engine # {selected_unit} Window</div>
            <div class="kpi-value">{len(engine_df)} <span class="kpi-unit">cycles</span></div>
            <div class="kpi-sub">Input sequence: last {min(len(engine_df), WINDOW_SIZE)} cycles</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 1.1rem;'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# 7. Interactive Visualization Tabs (Matched #111927 Plot Canvas)
# ---------------------------------------------------------
PLOT_BG = "#111927"
GRID_COLOR = "rgba(148, 163, 184, 0.14)"

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Uncertainty & Risk Diagnostics",
        "📈 Sensor Degradation & Heatmap",
        "🗂️ Fleet Summary & Raw Telemetry",
    ]
)

# --- TAB 1: Gauge + MC Dropout Posterior Distribution ---
with tab1:
    g_col, h_col = st.columns([1, 1.35])

    with g_col:
        st.subheader("RUL Health & Safety Threshold Gauge")
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=result["predicted_rul"],
                number={"suffix": " cycles", "font": {"size": 34, "color": "#F8FAFC"}},
                delta={
                    "reference": safety_threshold,
                    "increasing": {"color": "#10B981"},
                    "decreasing": {"color": "#EF4444"},
                    "suffix": " vs limit",
                },
                gauge={
                    "axis": {"range": [0, RUL_CAP], "tickwidth": 1, "tickcolor": "#94A3B8"},
                    "bar": {"color": "#00D4FF", "thickness": 0.28},
                    "bgcolor": PLOT_BG,
                    "borderwidth": 1,
                    "bordercolor": "#1F2D42",
                    "steps": [
                        {"range": [0, safety_threshold], "color": "rgba(239, 68, 68, 0.30)"},
                        {"range": [safety_threshold, safety_threshold * 2], "color": "rgba(245, 158, 11, 0.24)"},
                        {"range": [safety_threshold * 2, RUL_CAP], "color": "rgba(16, 185, 129, 0.20)"},
                    ],
                    "threshold": {
                        "line": {"color": "#EF4444", "width": 4},
                        "thickness": 0.82,
                        "value": safety_threshold,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            height=400,
            margin=dict(l=25, r=25, t=35, b=20),
            paper_bgcolor=PLOT_BG,
            font={"color": "#E2E8F0"},
        )
        st.plotly_chart(fig_gauge, width="stretch")

    with h_col:
        st.subheader(f"MC Dropout Predictive Distribution (Engine #{selected_unit})")
        fig_hist = go.Figure()

        fig_hist.add_vrect(
            x0=result["ci_95_lower"],
            x1=result["ci_95_upper"],
            fillcolor="rgba(0, 212, 255, 0.12)",
            layer="below",
            line_width=0,
            annotation_text="95% CI Region",
            annotation_position="top left",
            annotation_font_color="#7DD3FC",
        )

        fig_hist.add_trace(
            go.Histogram(
                x=result["mc_samples"],
                nbinsx=28,
                name="Stochastic Passes",
                marker=dict(color="#00D4FF", line=dict(color="#0284C7", width=1)),
                opacity=0.85,
            )
        )

        fig_hist.add_vline(
            x=result["predicted_rul"],
            line_dash="solid",
            line_width=2.5,
            line_color="#F8FAFC",
            annotation_text=f"Mean: {result['predicted_rul']}",
            annotation_position="top right",
            annotation_font_color="#F8FAFC",
        )
        fig_hist.add_vline(
            x=result["ci_95_lower"],
            line_dash="dash",
            line_width=2,
            line_color="#EF4444",
            annotation_text=f"Lower 95%: {result['ci_95_lower']}",
            annotation_position="bottom left",
            annotation_font_color="#FCA5A5",
        )
        fig_hist.add_vline(
            x=result["ci_95_upper"],
            line_dash="dash",
            line_width=2,
            line_color="#10B981",
            annotation_text=f"Upper 95%: {result['ci_95_upper']}",
            annotation_position="bottom right",
            annotation_font_color="#6EE7B7",
        )

        fig_hist.update_layout(
            xaxis_title="Predicted Remaining Useful Life (Cycles)",
            yaxis_title="MC Sample Frequency",
            height=400,
            margin=dict(l=20, r=20, t=35, b=20),
            paper_bgcolor=PLOT_BG,
            plot_bgcolor=PLOT_BG,
            font={"color": "#E2E8F0"},
            xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
            showlegend=False,
        )
        st.plotly_chart(fig_hist, width="stretch")


# --- TAB 2: Multi-Sensor Trends & 30-Cycle Degradation Heatmap ---
with tab2:
    scaled_view = pd.DataFrame(
        scaler.transform(engine_df[SELECTED_FEATURES]),
        columns=SELECTED_FEATURES,
    )
    scaled_view["time_cycles"] = engine_df["time_cycles"]
    window_view = scaled_view.tail(WINDOW_SIZE)

    s_col1, s_col2 = st.columns([1.25, 1])

    with s_col1:
        st.subheader("Normalized Sensor Trajectories (Last 30 Cycles)")
        default_sensors = [s for s in ["s_2", "s_3", "s_4", "s_11", "s_15"] if s in SELECTED_FEATURES]
        chosen_sensors = st.multiselect(
            "Select Sensors to Inspect",
            options=SELECTED_FEATURES,
            default=default_sensors[:4],
            format_func=lambda x: SENSOR_DESCRIPTIONS.get(x, x),
        )

        if chosen_sensors:
            fig_ts = go.Figure()
            palette = ["#00D4FF", "#F43F5E", "#10B981", "#F59E0B", "#A855F7", "#EAB308"]
            for idx, s in enumerate(chosen_sensors):
                fig_ts.add_trace(
                    go.Scatter(
                        x=window_view["time_cycles"],
                        y=window_view[s],
                        mode="lines+markers",
                        name=SENSOR_DESCRIPTIONS.get(s, s),
                        line=dict(width=2.5, color=palette[idx % len(palette)]),
                        marker=dict(size=5),
                    )
                )
            fig_ts.update_layout(
                xaxis_title="Flight Cycle",
                yaxis_title="Normalized Sensor Reading [0, 1]",
                height=410,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor=PLOT_BG,
                plot_bgcolor=PLOT_BG,
                font={"color": "#E2E8F0"},
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                xaxis=dict(gridcolor=GRID_COLOR),
                yaxis=dict(gridcolor=GRID_COLOR, range=[-0.05, 1.05]),
            )
            st.plotly_chart(fig_ts, width="stretch")

    with s_col2:
        st.subheader("30-Cycle All-Sensor Intensity Heatmap")
        sensor_only_cols = [c for c in SELECTED_FEATURES if c.startswith("s_")]
        z_matrix = window_view[sensor_only_cols].values.T

        fig_hm = go.Figure(
            data=go.Heatmap(
                z=z_matrix,
                x=window_view["time_cycles"].astype(str).tolist(),
                y=sensor_only_cols,
                colorscale="Tealrose",
                zmin=0,
                zmax=1,
                colorbar=dict(title="Scaled"),
            )
        )
        fig_hm.update_layout(
            xaxis_title="Flight Cycle",
            yaxis_title="Sensor ID",
            height=455,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor=PLOT_BG,
            plot_bgcolor=PLOT_BG,
            font={"color": "#E2E8F0"},
        )
        st.plotly_chart(fig_hm, width="stretch")


# --- TAB 3: Fleet Batch Screening & Raw Telemetry ---
with tab3:
    if full_df is not None and full_df["unit_id"].nunique() > 1:
        st.subheader("Fleet-Wide Risk Screening (All Uploaded Engines)")
        if st.button("Run Batch MC Dropout Screening Across All Engines"):
            fleet_rows = []
            progress = st.progress(0)
            all_units = sorted(full_df["unit_id"].unique().tolist())
            for idx, uid in enumerate(all_units):
                sub_df = full_df[full_df["unit_id"] == uid].sort_values("time_cycles")
                res_u = run_mc_dropout_inference(sub_df, n_iter=n_mc_passes, safety_threshold=safety_threshold)
                fleet_rows.append(
                    {
                        "Unit ID": uid,
                        "Cycles Logged": len(sub_df),
                        "Predicted Mean RUL": res_u["predicted_rul"],
                        "Uncertainty (±1σ)": res_u["uncertainty_std"],
                        "95% Lower Bound": res_u["ci_95_lower"],
                        "95% Upper Bound": res_u["ci_95_upper"],
                        "Risk Level": res_u["risk_level"],
                        "Maintenance Action": res_u["status"],
                    }
                )
                progress.progress((idx + 1) / len(all_units))

            fleet_summary = pd.DataFrame(fleet_rows).sort_values("95% Lower Bound")
            st.dataframe(fleet_summary, width="stretch")

            csv_bytes = fleet_summary.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Fleet Maintenance Report (.csv)",
                data=csv_bytes,
                file_name="FD001_Fleet_RUL_UQ_Report.csv",
                mime="text/csv",
            )
        st.markdown("---")

    st.subheader(f"Raw Sensor Telemetry — Engine #{selected_unit} (Last {WINDOW_SIZE} Cycles)")
    st.dataframe(engine_df.tail(WINDOW_SIZE), width="stretch")