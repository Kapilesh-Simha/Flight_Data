# ======================================================
# ✈️ X-PLANE PREDICTIVE MAINTENANCE STREAMLIT APP (Enhanced UI)
# ======================================================
import os
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model

# ---------- CONFIG / PATHS ----------
XGB_MODEL_PATH = r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\models\xplane_xgboost.pkl"
LSTM_MODEL_PATH = r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\models\xplane_lstm.h5"
SCALER_PATH = r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\models\lstm_scaler.pkl"
DATA_PATH = r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\data\processed\xplane_features.csv"
DEFAULT_LSTM_TIMESTEPS = 50
LOG_OUT_PATH = r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\logs\live_log.csv"

# ---------- APP CONFIG ----------
st.set_page_config(page_title="✈️ X-Plane Predictive Maintenance", layout="wide")

# ---------- CACHED HELPERS ----------
@st.cache_resource
def load_xgb_model(path=XGB_MODEL_PATH):
    if not os.path.exists(path):
        return None, 0.5
    data = joblib.load(path)
    if isinstance(data, dict):
        model = data.get("model", data.get("model_object", None)) or data
        threshold = data.get("threshold", 0.5)
    elif isinstance(data, (tuple, list)):
        model, threshold = data[0], data[1] if len(data) > 1 else 0.5
    else:
        model, threshold = data, 0.5
    return model, float(threshold)

@st.cache_resource
def load_lstm_model(path=LSTM_MODEL_PATH):
    if not os.path.exists(path):
        return None
    return load_model(path)

@st.cache_resource
def load_scaler(path=SCALER_PATH):
    if os.path.exists(path):
        return joblib.load(path)
    return None

# ---------- UTILITIES ----------
def live_stream(file_path=DATA_PATH):
    """Yield dataset rows one by one (simulate live)."""
    if not os.path.exists(file_path):
        return
    for row in pd.read_csv(file_path, chunksize=1):
        yield row

def clean_features_for_model(row_df, drop_cols=("failure",)):
    df = row_df.copy()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.select_dtypes(include=[np.number])

def sliding_windows(X, timesteps=50):
    Xs = []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i + timesteps])
    return np.stack(Xs, axis=0) if Xs else np.empty((0, timesteps, X.shape[1]))

# ---------- FEATURE IMPORTANCE ----------
def identify_top_contributors(xgb_model, scaler, features_df, top_k=3):
    if xgb_model is None or scaler is None:
        return None
    try:
        feat_names = list(xgb_model.feature_names_in_)
    except Exception:
        feat_names = None
    if feat_names is None:
        return None
    try:
        importances = getattr(xgb_model, "feature_importances_", None)
        if importances is None:
            importances = np.ones(len(feat_names))
    except Exception:
        importances = np.ones(len(feat_names))
    mean, scale = getattr(scaler, "mean_", None), getattr(scaler, "scale_", None)
    if mean is None or scale is None:
        return None
    row_vals = np.array([float(features_df.get(col, 0)) for col in feat_names])
    z = (row_vals - mean) / np.where(scale == 0, 1e-6, scale)
    scores = np.abs(z) * np.abs(importances)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [{"feature": feat_names[i], "value": row_vals[i], "score": scores[i]} for i in top_idx]

# ---------- STREAMLIT UI ----------
st.sidebar.header("Mode Selection")
mode = st.sidebar.radio("Choose mode", ["📡 Real-Time Streaming", "📊 Interactive Batch Analysis"])

xgb_model, saved_threshold = load_xgb_model()
lstm_model = load_lstm_model()
scaler = load_scaler()

# ---------- REAL-TIME MODE ----------
if mode == "📡 Real-Time Streaming":
    st.title("📡 Real-Time Predictive Maintenance Dashboard")

    # Sidebar controls
    st.sidebar.subheader("🔧 Stream Controls")
    refresh_rate = st.sidebar.slider("Refresh Interval (seconds)", 0.5, 10.0, 1.0, 0.5)
    start_stream = st.sidebar.button("▶ Start Live Streaming")
    stop_stream = st.sidebar.button("■ Stop Live Streaming")

    st.sidebar.subheader("🎯 Risk Zone Thresholds")
    green_threshold = st.sidebar.slider("🟢 Green zone upper limit", 0.0, 1.0, 0.5, 0.01)
    yellow_threshold = st.sidebar.slider("🟡 Yellow zone upper limit", green_threshold, 1.0, 0.75, 0.01)

    # Layout: Left gauge, Right status panel
    col_left, col_right = st.columns([2, 1])
    with col_left:
        gauge_ph = st.empty()
        chart_xgb = st.line_chart(pd.DataFrame(columns=["xgb_prob"]))
        chart_lstm = st.line_chart(pd.DataFrame(columns=["lstm_prob"]))
    with col_right:
        status_area = st.empty()
        faulty_area = st.empty()

    # State management
    if "stream_running" not in st.session_state:
        st.session_state.stream_running = False

    # ---------- DYNAMIC GAUGE ----------
    def render_gauge(prob, g_thresh, y_thresh):
        """Dynamic Plotly gauge with color and background glow."""
        prob = float(np.clip(prob, 0.0, 1.0))
        if prob <= g_thresh:
            bar_color, glow_color = "green", "rgba(0,255,0,0.25)"
        elif prob <= y_thresh:
            bar_color, glow_color = "gold", "rgba(255,215,0,0.25)"
        else:
            bar_color, glow_color = "red", "rgba(255,0,0,0.25)"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Failure Probability", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': bar_color},
                'steps': [
                    {'range': [0, g_thresh], 'color': "rgba(0,255,0,0.25)"},
                    {'range': [g_thresh, y_thresh], 'color': "rgba(255,215,0,0.25)"},
                    {'range': [y_thresh, 1.0], 'color': "rgba(255,0,0,0.25)"}
                ],
                'threshold': {'line': {'color': bar_color, 'width': 3}, 'value': prob}
            }
        ))
        fig.update_layout(
            height=260, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor=glow_color,
            transition={'duration': 500, 'easing': 'cubic-in-out'}
        )
        gauge_ph.plotly_chart(fig, use_container_width=True, key=f"gauge_{time.time_ns()}")

    # ---------- ZONE LABEL ----------
    def zone_label(prob, g_thresh, y_thresh):
        if prob <= g_thresh:
            return "🟢 STABLE", "green", "Engine is operating normally."
        elif prob <= y_thresh:
            return "🟡 LOW RISK", "gold", "Minor anomalies detected. Monitor closely."
        else:
            return "🔴 HIGH RISK", "red", "Potential fault detected! Immediate inspection advised."

    # ---------- STREAM LOOP ----------
    if start_stream:
        st.session_state.stream_running = True
    if stop_stream:
        st.session_state.stream_running = False

    if st.session_state.stream_running:
        last_combined = 0.0
        for row in live_stream():
            if not st.session_state.stream_running:
                break
            features = clean_features_for_model(row)
            # Predictions
            try:
                xgb_prob = float(xgb_model.predict_proba(features)[0][1])
            except Exception:
                xgb_prob = 0.0
            try:
                scaled = scaler.transform(features)
                if "seq_buf" not in st.session_state:
                    st.session_state.seq_buf = []
                st.session_state.seq_buf.append(scaled.flatten())
                if len(st.session_state.seq_buf) >= DEFAULT_LSTM_TIMESTEPS:
                    X_seq = np.array(st.session_state.seq_buf[-DEFAULT_LSTM_TIMESTEPS:]).reshape(1, DEFAULT_LSTM_TIMESTEPS, features.shape[1])
                    lstm_prob = float(lstm_model.predict(X_seq, verbose=0)[0][0])
                else:
                    lstm_prob = 0.0
            except Exception:
                lstm_prob = 0.0

            combined_prob = max(xgb_prob, lstm_prob)
            smooth = last_combined + (combined_prob - last_combined)
            last_combined = smooth

            render_gauge(smooth, green_threshold, yellow_threshold)
            chart_xgb.add_rows(pd.DataFrame({"xgb_prob": [xgb_prob]}))
            chart_lstm.add_rows(pd.DataFrame({"lstm_prob": [lstm_prob]}))

            zone_txt, color, desc = zone_label(smooth, green_threshold, yellow_threshold)
            status_area.markdown(
                f"""
                <div style="
                    padding:12px;
                    border-radius:12px;
                    background:rgba(255,255,255,0.05);
                    border-left:6px solid {color};
                    box-shadow:0 0 25px {color}80;
                ">
                <h3 style="margin:0;color:{color};font-size:22px">{zone_txt}</h3>
                <p style="margin:4px 0;font-size:16px;color:white">{desc}</p>
                <p style="margin:4px 0;color:lightgray">
                    Combined Failure Probability: <b style="color:{color}">{smooth:.3f}</b>
                </p>
                </div>
                """, unsafe_allow_html=True
            )

            top_feats = identify_top_contributors(xgb_model, scaler, features)
            if top_feats:
                faulty_area.markdown(
                    "<b>Likely Fault Contributors:</b><br>" +
                    "<br>".join([f"{f['feature']}: {f['value']:.2f}" for f in top_feats]),
                    unsafe_allow_html=True
                )
            else:
                faulty_area.info("No significant contributors identified yet.")
            time.sleep(refresh_rate)

        st.info("Stream stopped.")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("🛫 Developed for predictive maintenance visualization. Combines XGBoost & LSTM predictions with live telemetry.")