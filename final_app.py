# main_app.py
# ======================================================
# ✈️ X-PLANE PREDICTIVE MAINTENANCE STREAMLIT APP (improved)
# ======================================================
import os
import time
from datetime import datetime
import io

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model

# ---------- CONFIG / PATHS ----------
XGB_MODEL_PATH = r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\models\xplane_xgboost.pkl"
LSTM_MODEL_PATH = r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\models\xplane_lstm.h5"
SCALER_PATH = r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\models\lstm_scaler.pkl"
DATA_PATH = r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\data\processed\xplane_features.csv"
DEFAULT_LSTM_TIMESTEPS = 50
LOG_OUT_PATH = r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\logs\live_log.csv"  # where to persist logs

# ---------- APP CONFIG (must be first Streamlit command) ----------
st.set_page_config(page_title="✈️ X-Plane Predictive Maintenance", layout="wide")

# ---------- CACHED HELPERS ----------
@st.cache_resource
def load_xgb_model(path=XGB_MODEL_PATH):
    if not os.path.exists(path):
        return None, 0.5
    data = joblib.load(path)
    # support dictionary or (model, threshold) or direct model pickles
    if isinstance(data, dict):
        model = data.get("model", data.get("model_object", None)) or data
        threshold = data.get("threshold", 0.5)
    elif isinstance(data, (tuple, list)):
        try:
            model, threshold = data[0], data[1]
        except Exception:
            model, threshold = data, 0.5
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
    """Convert row (1-row dataframe) to model-ready numeric DataFrame."""
    df = row_df.copy()
    # drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    # drop explicitly
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])
    # coerce to numeric where possible
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # keep numeric only
    df = df.select_dtypes(include=[np.number])
    return df


def sliding_windows(X, timesteps=50):
    Xs = []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i + timesteps])
    if len(Xs) == 0:
        return np.empty((0, timesteps, X.shape[1]))
    return np.stack(Xs, axis=0)


def plot_confusion(cm, labels=["0", "1"], title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_roc(y_true, y_proba, label="Model"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    return fig, roc_auc


def identify_top_contributors(xgb_model, scaler, features_df, top_k=3):
    """
    Heuristic: use XGBoost feature_importances_ and standardized deviation (z = (val - mean)/scale)
    Multiply absolute z by importance to get a score; return top_k features (name, score, value).
    Returns None if we lack model/scaler info.
    """
    if xgb_model is None or scaler is None:
        return None

    # feature names
    try:
        feat_names = list(xgb_model.feature_names_in_)
    except Exception:
        try:
            feat_names = xgb_model.get_booster().feature_names
        except Exception:
            feat_names = None

    if feat_names is None:
        return None

    # get importance and align
    try:
        importances = getattr(xgb_model, "feature_importances_", None)
        if importances is None:
            # fallback: XGBoost booster
            try:
                importances = np.array(xgb_model.get_booster().get_score(importance_type="weight").values(), dtype=float)
            except Exception:
                importances = np.ones(len(feat_names))
    except Exception:
        importances = np.ones(len(feat_names))

    # scaler mean and scale
    mean = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)
    if mean is None or scale is None:
        return None

    # features_df may be a subset of features; map values to feat_names
    row_vals = np.zeros(len(feat_names))
    for i, name in enumerate(feat_names):
        if name in features_df.columns:
            row_vals[i] = float(features_df[name].values[0])
        else:
            row_vals[i] = 0.0

    # compute standardized deviation (avoid divide by zero)
    scale_safe = np.where(scale == 0, 1e-6, scale)
    z = (row_vals - mean) / scale_safe
    scores = np.abs(z) * np.abs(importances)
    # get top indices
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_idx:
        results.append({
            "feature": feat_names[idx],
            "value": row_vals[idx],
            "z": float(z[idx]),
            "importance": float(importances[idx]) if len(importances) == len(feat_names) else None,
            "score": float(scores[idx])
        })
    return results


# ---------- UI / App ----------
st.sidebar.header("Mode Selection")
mode = st.sidebar.radio("Choose mode", ["📡 Real-Time Streaming", "📊 Interactive Batch Analysis"])

with st.sidebar.expander("📘 About This Dashboard (Quick)"):
    st.markdown(
        """
        **Real-time predictive maintenance** demo using XGBoost + LSTM.
        - Upload a processed CSV (optional) for batch mode.
        - Use *Real-Time* mode to simulate live telemetry from `data/processed/xplane_features.csv`.
        """
    )

# Load models (cached)
xgb_model, saved_threshold = load_xgb_model()
lstm_model = load_lstm_model()
scaler = load_scaler()

# ---------- Shared sidebar controls ----------
st.sidebar.markdown("---")
st.sidebar.subheader("Logging & Export")
do_logging = st.sidebar.checkbox("Enable Live Logging (in-memory)", value=False)
persist_logs_disk = st.sidebar.checkbox("Persist logs to disk (append)", value=False)
if persist_logs_disk:
    os.makedirs(os.path.dirname(LOG_OUT_PATH), exist_ok=True)
st.sidebar.markdown("---")

# ---------- Real-time mode ----------
if mode == "📡 Real-Time Streaming":
    st.title("📡 Real-Time Predictive Maintenance Dashboard")
    st.sidebar.subheader("🔧 Stream Controls")
    refresh_rate = st.sidebar.slider("Refresh Interval (seconds)", 0.5, 10.0, 1.0, 0.5)
    start_stream = st.sidebar.button("▶ Start Live Streaming")
    stop_stream = st.sidebar.button("■ Stop Live Streaming")

    # Risk zone sliders
    st.sidebar.subheader("🎯 Risk Zone Thresholds")
    green_threshold = st.sidebar.slider("🟢 Green zone upper limit", 0.0, 1.0, 0.5, 0.01)
    yellow_threshold = st.sidebar.slider("🟡 Yellow zone upper limit", green_threshold, 1.0, 0.75, 0.01)
    red_threshold = 1.0

    # Logging controls
    st.sidebar.subheader("📥 Live Log Controls")
    log_button = st.sidebar.button("Start/Stop Logging (toggle)")
    download_logs_button = st.sidebar.button("Download Log CSV")

    # instantiate in-memory log storage in session state
    if "live_log_df" not in st.session_state:
        st.session_state.live_log_df = pd.DataFrame(
            columns=[
                "timestamp", "xgb_prob", "lstm_prob", "combined_prob", "meter_zone"
            ]
        )
        st.session_state.logging_enabled = False

    # toggle logging when user presses log_button
    if log_button:
        st.session_state.logging_enabled = not st.session_state.logging_enabled
        st.success("Logging turned ON" if st.session_state.logging_enabled else "Logging turned OFF")

    # Download logs if requested
    if download_logs_button:
        if st.session_state.live_log_df.shape[0] == 0:
            st.warning("No log rows yet.")
        else:
            csv_bytes = st.session_state.live_log_df.to_csv(index=False).encode("utf-8")
            st.sidebar.download_button("Download logs (CSV)", data=csv_bytes, file_name="live_log.csv", mime="text/csv")

    # Layout: left gauge + charts, right status panel
    col_left, col_right = st.columns([2, 1])
    with col_left:
        gauge_ph = st.empty()
        chart_xgb = st.line_chart(pd.DataFrame(columns=["xgb_prob"]))
        chart_lstm = st.line_chart(pd.DataFrame(columns=["lstm_prob"]))
    with col_right:
        status_area = st.empty()
        faulty_area = st.empty()

    # internal stop flag via session_state
    if "stream_running" not in st.session_state:
        st.session_state.stream_running = False

    def render_gauge(prob, g_thresh, y_thresh):
        """Plotly gauge with colored steps."""
        prob = float(np.clip(prob, 0.0, 1.0))
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Failure Probability", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, g_thresh], 'color': "lightgreen"},
                    {'range': [g_thresh, y_thresh], 'color': "orange"},
                    {'range': [y_thresh, 1.0], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 3},
                    'thickness': 0.8,
                    'value': prob
                }
            }
        ))
        fig.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10))
        gauge_ph.plotly_chart(fig, use_container_width=True)

    def zone_label(prob, g_thresh, y_thresh):
        if prob <= g_thresh:
            return "🟢 STABLE"
        if prob <= y_thresh:
            return "🟡 LOW RISK"
        return "🔴 HIGH RISK"

    # Run the live simulation in a loop until stop_stream is pressed.
    # We'll check session_state to manage starting/stopping reliably.
    if start_stream:
        st.session_state.stream_running = True

    if stop_stream:
        st.session_state.stream_running = False

    # main streaming loop — implemented as a generator consumption inside Streamlit's runtime
    if st.session_state.stream_running:
        # Use a container to avoid appending indefinitely in page output
        stream_container = st.empty()
        stream_placeholder = stream_container.container()

        # For smoothing gauge
        last_combined = 0.0

        # Iterate through rows
        for row in live_stream():
            if not st.session_state.stream_running:
                break  # exit loop if user clicked Stop
            try:
                features = clean_features_for_model(row, drop_cols=("failure",))
                # extract key telemetry for display (safe extraction)
                engine_rpm = float(row['rpm_1engin'].values[0]) if 'rpm_1engin' in row.columns else 0.0
                n1 = float(row['N1__1_pcnt'].values[0]) if 'N1__1_pcnt' in row.columns else 0.0
                n2 = float(row['N1__2_pcnt'].values[0]) if 'N1__2_pcnt' in row.columns else 0.0
                egt1 = float(row['EGT_1__deg'].values[0]) if 'EGT_1__deg' in row.columns else 0.0
                oil_temp1 = float(row['OILT1__deg'].values[0]) if 'OILT1__deg' in row.columns else 0.0
                fuel_pressure = float(row['FUEP1__psi'].values[0]) if 'FUEP1__psi' in row.columns else 0.0
            except Exception as e:
                # skip broken row
                st.error(f"Row parse error: {e}")
                continue

            # XGBoost inference
            try:
                if xgb_model is not None and features.shape[1] > 0:
                    # align feature order if model expects specific features
                    try:
                        expected = list(xgb_model.feature_names_in_)
                        for col in expected:
                            if col not in features.columns:
                                features[col] = 0.0
                        features = features[expected]
                    except Exception:
                        pass
                    xgb_prob = float(xgb_model.predict_proba(features)[0][1])
                else:
                    xgb_prob = 0.0
            except Exception:
                xgb_prob = 0.0

            # LSTM inference
            try:
                if lstm_model is not None and scaler is not None and features.shape[1] > 0:
                    scaled = scaler.transform(features)
                    # build sequence buffer in session_state for continuity
                    if "seq_buffer" not in st.session_state:
                        st.session_state.seq_buffer = []
                    st.session_state.seq_buffer.append(scaled.flatten())
                    if len(st.session_state.seq_buffer) >= DEFAULT_LSTM_TIMESTEPS:
                        X_seq = np.array(st.session_state.seq_buffer[-DEFAULT_LSTM_TIMESTEPS:]).reshape(1, DEFAULT_LSTM_TIMESTEPS, features.shape[1])
                        lstm_prob = float(lstm_model.predict(X_seq, verbose=0)[0][0])
                    else:
                        lstm_prob = 0.0
                else:
                    lstm_prob = 0.0
            except Exception:
                lstm_prob = 0.0

            combined_prob = max(xgb_prob, lstm_prob)
            # smoothing
            smooth_combined = last_combined + (combined_prob - last_combined) * 1.0
            last_combined = smooth_combined

            # render gauge & update charts
            render_gauge(smooth_combined, green_threshold, yellow_threshold)
            chart_xgb.add_rows(pd.DataFrame({"xgb_prob": [xgb_prob]}))
            chart_lstm.add_rows(pd.DataFrame({"lstm_prob": [lstm_prob]}))

            # right side status
            zone_txt = zone_label(smooth_combined, green_threshold, yellow_threshold)
            status_area.markdown(
                f"""
                <div style="padding:8px;border-radius:6px;border:1px solid #eee">
                <h3 style="margin:0">{zone_txt}</h3>
                <p style="margin:4px 0">Combined failure probability: <b>{smooth_combined:.3f}</b></p>
                <p style="margin:4px 0">Engine RPM: <b>{engine_rpm:.1f}</b> | N1: <b>{n1:.1f}</b> | N2: <b>{n2:.1f}</b></p>
                <p style="margin:4px 0">EGT1: <b>{egt1:.1f}°C</b> | Oil T1: <b>{oil_temp1:.1f}°C</b> | Fuel P: <b>{fuel_pressure:.2f} psi</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Faulty part heuristic (if models / scaler exist)
            contributors = identify_top_contributors(xgb_model, scaler, features, top_k=3)
            if contributors:
                bad_html = "<b>Top contributor signals (heuristic):</b><br>"
                for c in contributors:
                    bad_html += f"{c['feature']}: value={c['value']:.3f}, z={c['z']:.2f}, score={c['score']:.3f}<br>"
                faulty_area.markdown(bad_html, unsafe_allow_html=True)
            else:
                faulty_area.info("Fault contributor info not available (model/scaler missing).")

            # Logging
            if do_logging and st.session_state.logging_enabled:
                row_log = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "xgb_prob": xgb_prob,
                    "lstm_prob": lstm_prob,
                    "combined_prob": smooth_combined,
                    "meter_zone": zone_txt
                }
                st.session_state.live_log_df = pd.concat([st.session_state.live_log_df, pd.DataFrame([row_log])], ignore_index=True)
                if persist_logs_disk:
                    # append to disk
                    try:
                        header = not os.path.exists(LOG_OUT_PATH)
                        st.session_state.live_log_df.tail(1).to_csv(LOG_OUT_PATH, mode="a", header=header, index=False)
                    except Exception as e:
                        st.warning(f"Could not persist log to disk: {e}")

            # small delay
            time.sleep(refresh_rate)

        # end streaming loop
        st.info("Stream stopped.")

# ---------- Batch analysis mode ----------
if mode == "📊 Interactive Batch Analysis":
    st.title("📊 Interactive Batch Analysis")
    uploaded = st.file_uploader("Upload X-Plane Processed CSV (csv)", type=["csv"])
    model_choice = st.selectbox("Choose model for analysis", ["XGBoost", "LSTM", "Both"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")
            df = None

        if df is not None:
            st.success(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")
            st.dataframe(df.head(50))

            if model_choice in ("XGBoost", "Both"):
                st.subheader("XGBoost Analysis")
                X = clean_features_for_model(df.drop(columns=["failure"], errors="ignore"))
                y = df["failure"].values if "failure" in df.columns else None
                try:
                    if xgb_model is None:
                        st.warning("XGBoost model not found. Place a model in models/ and reload.")
                    else:
                        # align to model's expected features if available
                        try:
                            expected = list(xgb_model.feature_names_in_)
                            for col in expected:
                                if col not in X.columns:
                                    X[col] = 0.0
                            X = X[expected]
                        except Exception:
                            pass
                        proba = xgb_model.predict_proba(X)[:, 1]
                        preds = (proba >= saved_threshold).astype(int)
                        out = df.copy()
                        out["xgb_proba"] = proba
                        out["xgb_pred"] = preds
                        st.dataframe(out.head(50))
                        if y is not None:
                            cm = confusion_matrix(y, preds)
                            st.pyplot(plot_confusion(cm, ["No Failure", "Failure"], "XGBoost Confusion Matrix"))
                            fig_roc, auc_val = plot_roc(y, proba, "XGBoost")
                            st.pyplot(fig_roc)
                            st.success(f"ROC-AUC: {auc_val:.3f}")
                except Exception as e:
                    st.error(f"XGBoost inference failed: {e}")

            if model_choice in ("LSTM", "Both"):
                st.subheader("LSTM Analysis")
                df_num = df.select_dtypes(include=[np.number]).drop(columns=["failure"], errors="ignore")
                y = df["failure"].values if "failure" in df.columns else None
                try:
                    if lstm_model is None or scaler is None:
                        st.warning("LSTM model or scaler missing - place them in models/ and reload.")
                    else:
                        X_scaled = scaler.transform(df_num)
                        X_seq = sliding_windows(X_scaled, timesteps=DEFAULT_LSTM_TIMESTEPS)
                        if X_seq.shape[0] == 0:
                            st.warning("Not enough rows to build LSTM sequences with current timesteps.")
                        else:
                            proba = lstm_model.predict(X_seq).ravel()
                            preds = (proba >= 0.5).astype(int)
                            out = pd.DataFrame({"lstm_proba": proba, "lstm_pred": preds})
                            st.dataframe(out.head(50))
                            if y is not None:
                                y_true = y[DEFAULT_LSTM_TIMESTEPS: DEFAULT_LSTM_TIMESTEPS + len(proba)]
                                cm = confusion_matrix(y_true, preds)
                                st.pyplot(plot_confusion(cm, ["No Failure", "Failure"], "LSTM Confusion Matrix"))
                                fig_roc, auc_val = plot_roc(y_true, proba, "LSTM")
                                st.pyplot(fig_roc)
                                st.success(f"ROC-AUC: {auc_val:.3f}")
                except Exception as e:
                    st.error(f"LSTM inference failed: {e}")

# ---------- Footer: allow exporting logs if any ----------
st.markdown("---")
if "live_log_df" in st.session_state and st.session_state.live_log_df.shape[0] > 0:
    st.write(f"Live log rows: {st.session_state.live_log_df.shape[0]}")
    csv_bytes = st.session_state.live_log_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download current in-memory log (CSV)", csv_bytes, "live_log.csv", "text/csv")

st.caption("Tip: Save models in `models/` and processed CSV in `data/processed/`. Fault contributor output is a heuristic (importance × standardised deviation).")
