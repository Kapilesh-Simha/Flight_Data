# main_app.py
import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_score, recall_score, f1_score
)
from tensorflow.keras.models import load_model

# ------------------ CONFIG ------------------
st.set_page_config(page_title="✈️ X-Plane Predictive Maintenance", layout="wide")

# Paths - update to your environment
XGB_MODEL_PATH = r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\models\xplane_xgboost.pkl"
LSTM_MODEL_PATH = r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\models\xplane_lstm.h5"
SCALER_PATH = r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\models\lstm_scaler.pkl"
DATA_PATH = r"C:\Users\kapil\OneDrive\Desktop\xplane_predictive_project\data\processed\xplane_features.csv"
DEFAULT_LSTM_TIMESTEPS = 50

# ------------------ UTIL HELPERS ------------------
@st.cache_resource
def load_xgb_model(path=XGB_MODEL_PATH):
    if not os.path.exists(path):
        return None, 0.5
    data = joblib.load(path)
    # support saved (model,threshold) dict/tuple or plain model
    if isinstance(data, dict):
        model = data.get("model", data.get("model_object", data))
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

def clean_tabular_for_xgb(df, model=None):
    """Drop unnamed columns, coerce objects to numeric and align to model features if available"""
    df = df.copy()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    for c in df.select_dtypes(include=["object"]).columns:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            pass
    df = df.select_dtypes(include=[np.number])
    if model is not None:
        expected = None
        try:
            expected = list(model.feature_names_in_)
        except Exception:
            expected = None
        if expected:
            for col in expected:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[expected]
    return df

def sliding_windows(X, timesteps=50):
    Xs = []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i+timesteps])
    if len(Xs) == 0:
        return np.empty((0, timesteps, X.shape[1]))
    return np.stack(Xs, axis=0)

# ------------------ PLOT HELPERS for batch mode ------------------
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve as sk_roc_curve, auc as sk_auc

def plot_confusion(cm, labels=["0","1"], title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4,4))
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
    fpr, tpr, _ = sk_roc_curve(y_true, y_proba)
    roc_auc = sk_auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc:.2f})")
    ax.plot([0,1], [0,1], color="grey", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    return fig, roc_auc

# ------------------ STREAM DATA SIMULATOR ------------------
def live_stream(file_path=DATA_PATH):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No data file at {file_path}")
    df_iter = pd.read_csv(file_path, chunksize=1)
    for row in df_iter:
        yield row

# ------------------ LOAD MODELS ------------------
xgb_model, saved_threshold = load_xgb_model()
lstm_model = load_lstm_model()
scaler = load_scaler()

# ------------------ APP UI ------------------
st.sidebar.header("Mode & Controls")
mode = st.sidebar.radio("Choose mode", ["📡 Real-Time Streaming", "📊 Interactive Batch Analysis"])
# thresholds for gauge coloring
green_threshold = st.sidebar.slider("Green upper bound", 0.0, 1.0, 0.20, 0.01)
yellow_threshold = st.sidebar.slider("Yellow upper bound", 0.0, 1.0, 0.60, 0.01)
if green_threshold >= yellow_threshold:
    st.sidebar.warning("Green must be < Yellow — auto-adjusting Yellow.")
    yellow_threshold = min(green_threshold + 0.1, 1.0)

# ------------------ REAL-TIME MODE ------------------
if mode == "📡 Real-Time Streaming":
    st.title("📡 Real-Time Predictive Maintenance Dashboard")

    refresh_rate = st.sidebar.slider("Refresh Interval (seconds)", 1, 10, 2)
    failure_threshold = st.sidebar.slider("Alert threshold (either model)", 0.0, 1.0, 0.5, 0.01)
    start_stream = st.sidebar.button("▶ Start Live Streaming")

    # Create the fixed UI placeholders ONCE
    col_left, col_right = st.columns([2,1])
    # gauges row: two gauges side-by-side; each gauge has a small status column to its right
    col_gx, col_gl = col_left.columns(2)
    # For XGB: in column 1 create two subcolumns (gauge and text)
    gx_gauge_col, gx_text_col = col_gx.columns([3, 1])
    gl_gauge_col, gl_text_col = col_gl.columns([3, 1])

    xgb_gauge_ph = gx_gauge_col.empty()
    xgb_text_ph = gx_text_col.empty()

    lstm_gauge_ph = gl_gauge_col.empty()
    lstm_text_ph = gl_text_col.empty()

    # Live line charts and engine status area (below gauges)
    chart_col = col_right
    chart_xgb = chart_col.empty()
    chart_lstm = chart_col.empty()  # we will reassign a single combined area below
    # create separate line charts left in the page as well to avoid re-creating them
    # (we'll use them inside the loop by referencing the Chart object)
    xgb_line = st.line_chart([], height=150)
    lstm_line = st.line_chart([], height=150)

    # Global alert placeholder updated in-place
    alert_ph = st.empty()
    # small telemetry box
    telemetry_ph = st.empty()

    # smoothing helper
    last_xgb = [0.0]
    last_lstm = [0.0]
    TIMESTEPS = DEFAULT_LSTM_TIMESTEPS
    seq_buffer = []

    def build_gauge_figure(value, green=green_threshold, yellow=yellow_threshold, label="Failure Probability"):
        """Return a Plotly gauge figure for a 0..1 value"""
        v = max(0.0, min(1.0, float(value)))
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=v,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': label, 'font': {'size': 16}},
            number={'valueformat': ".3f"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, green], 'color': 'lightgreen'},
                    {'range': [green, yellow], 'color': 'yellow'},
                    {'range': [yellow, 1.0], 'color': 'salmon'}
                ],
                'threshold': {
                    'line': {'color': 'black', 'width': 3},
                    'thickness': 0.75,
                    'value': v
                }
            }
        ))
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=220)
        return fig

    def status_text_and_color(prob, g=green_threshold, y=yellow_threshold):
        p = max(0.0, min(1.0, float(prob)))
        if p < g:
            return ("🟢 STABLE", "green", "System operating normally.")
        if p < y:
            return ("🟡 LOW RISK", "orange", "Minor anomalies detected — monitor closely.")
        return ("🔴 HIGH RISK", "red", "Potential failure detected — immediate attention required!")

    def safe_predict_xgb(single_row_df):
        try:
            X_tab = clean_tabular_for_xgb(single_row_df, model=xgb_model) if xgb_model else single_row_df.select_dtypes(include=[np.number])
            if X_tab.shape[1] == 0:
                return 0.0
            # If scaler exists for XGB use it else pass raw
            proba = xgb_model.predict_proba(X_tab)[0][1]
            return float(proba)
        except Exception:
            return 0.0

    def safe_predict_lstm(single_row_df):
        try:
            # numeric only and maintain column order
            num = single_row_df.select_dtypes(include=[np.number])
            if num.shape[1] == 0:
                return 0.0
            if scaler is not None:
                scaled = scaler.transform(num.values)  # shape (1, n_features)
            else:
                # fallback: standard scale by row (not ideal but prevents crash)
                scaled = StandardScaler().fit_transform(num.values)
            # append to buffer
            seq_buffer.append(scaled.ravel())
            if len(seq_buffer) >= TIMESTEPS and lstm_model is not None:
                X_seq = np.array(seq_buffer[-TIMESTEPS:]).reshape(1, TIMESTEPS, scaled.shape[1])
                p = float(lstm_model.predict(X_seq, verbose=0)[0][0])
                return p
            return 0.0
        except Exception:
            return 0.0

    # Run loop only when user clicks the sidebar Start button
    if start_stream:
        try:
            for row in live_stream():
                # ensure row is a DataFrame (chunksize=1 yields DataFrame)
                features_df = row.copy()
                # compute telemetry values (safe access)
                telemetry = {
                    "rpm_1engin": float(row.get('rpm_1engin', [0])[0]) if 'rpm_1engin' in row else 0.0,
                    "N1__1_pcnt": float(row.get('N1__1_pcnt', [0])[0]) if 'N1__1_pcnt' in row else 0.0,
                    "EGT_1__deg": float(row.get('EGT_1__deg', [0])[0]) if 'EGT_1__deg' in row else 0.0,
                    "OILT1__deg": float(row.get('OILT1__deg', [0])[0]) if 'OILT1__deg' in row else 0.0,
                }

                # XGBoost prediction
                xgb_prob = 0.0
                if xgb_model is not None:
                    try:
                        # align and predict; use clean_tabular_for_xgb to align columns
                        X_tab = clean_tabular_for_xgb(features_df, model=xgb_model)
                        if X_tab.shape[1] > 0:
                            xgb_prob = float(xgb_model.predict_proba(X_tab)[0][1])
                    except Exception:
                        xgb_prob = safe_predict_xgb(features_df)

                # LSTM prediction
                lstm_prob = 0.0
                if lstm_model is not None:
                    lstm_prob = safe_predict_lstm(features_df)

                # Smooth values visually
                last_xgb[0] = last_xgb[0] + (xgb_prob - last_xgb[0]) * 0.4
                last_lstm[0] = last_lstm[0] + (lstm_prob - last_lstm[0]) * 0.4
                vis_xgb = last_xgb[0]
                vis_lstm = last_lstm[0]

                # Update telemetry area (in place)
                telemetry_ph.markdown(
                    f"**RPM:** {telemetry['rpm_1engin']:.1f} &nbsp;&nbsp; **N1:** {telemetry['N1__1_pcnt']:.2f}%  \n"
                    f"**EGT:** {telemetry['EGT_1__deg']:.1f}°C &nbsp;&nbsp; **Oil Temp:** {telemetry['OILT1__deg']:.1f}°C"
                )

                # Update XGB gauge + status (in place)
                fig_xgb = build_gauge_figure(vis_xgb, green=green_threshold, yellow=yellow_threshold, label="XGBoost Failure Prob")
                xgb_gauge_ph.plotly_chart(fig_xgb, use_container_width=True, key="xgb_gauge")
                s_text, s_color, s_desc = status_text_and_color(vis_xgb, g=green_threshold, y=yellow_threshold)
                xgb_text_ph.markdown(f"<h3 style='color:{s_color}; margin-top:55px'>{s_text}</h3><div style='font-size:14px'>{s_desc}</div>", unsafe_allow_html=True)

                # Update LSTM gauge + status (in place)
                fig_lstm = build_gauge_figure(vis_lstm, green=green_threshold, yellow=yellow_threshold, label="LSTM Failure Prob")
                lstm_gauge_ph.plotly_chart(fig_lstm, use_container_width=True, key="lstm_gauge")
                l_text, l_color, l_desc = status_text_and_color(vis_lstm, g=green_threshold, y=yellow_threshold)
                lstm_text_ph.markdown(f"<h3 style='color:{l_color}; margin-top:55px'>{l_text}</h3><div style='font-size:14px'>{l_desc}</div>", unsafe_allow_html=True)

                # Append to small live-line charts (these use the same chart objects created once)
                xgb_line.add_rows({"XGBoost": [vis_xgb]})
                lstm_line.add_rows({"LSTM": [vis_lstm]})

                # Global alert — update in-place (do not create new elements each loop)
                if (xgb_prob >= failure_threshold) or (lstm_prob >= failure_threshold):
                    alert_ph.error(f"🚨 HIGH FAILURE RISK! XGB={xgb_prob:.3f} LSTM={lstm_prob:.3f}")
                else:
                    alert_ph.success(f"✅ Stable | XGB={xgb_prob:.3f} LSTM={lstm_prob:.3f}")

                # sleep for refresh
                time.sleep(refresh_rate)
        except Exception as e:
            # If the loop crashes show helpful error in the alert placeholder
            alert_ph.error(f"Stream stopped due to error: {e}")

# ------------------ INTERACTIVE BATCH ANALYSIS ------------------
if mode == "📊 Interactive Batch Analysis":
    st.title("📊 Interactive Batch Analysis")

    uploaded = st.file_uploader("Upload X-Plane Processed CSV", type=["csv"])
    model_choice = st.selectbox("Select Model", ["XGBoost", "LSTM", "Both"], index=0)

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"✅ Loaded file with {df.shape[0]} rows and {df.shape[1]} columns")
        st.dataframe(df.head())

        # ---------------- MODEL RUN ----------------
        if model_choice in ["XGBoost", "Both"]:
            st.subheader("XGBoost Analysis")
            X = df.drop(columns=["failure", "Unnamed: 34"], errors="ignore")
            y = df["failure"] if "failure" in df.columns else None

            try:
                X_tab = clean_tabular_for_xgb(X, model=xgb_model) if xgb_model else X.select_dtypes(include=[np.number])
                proba = xgb_model.predict_proba(X_tab)[:, 1]
                preds = (proba >= 0.5).astype(int)

                out = X_tab.copy()
                out["failure_proba"] = proba
                out["failure_pred"] = preds
                st.dataframe(out.head(50))

                if y is not None:
                    cm = confusion_matrix(y[:len(preds)], preds)
                    st.pyplot(plot_confusion(cm, title="XGBoost Confusion"))
                    fig_roc, auc_val = plot_roc(y[:len(proba)], proba, label="XGBoost")
                    st.pyplot(fig_roc)
                    st.success(f"ROC-AUC: {auc_val:.3f}")
            except Exception as e:
                st.error(f"XGBoost inference failed: {e}")

        if model_choice in ["LSTM", "Both"]:
            st.subheader("LSTM Analysis")
            df_num = df.select_dtypes(include=[np.number]).drop(columns=["failure"], errors="ignore")
            y = df["failure"] if "failure" in df.columns else None

            try:
                if scaler is None:
                    scaler_local = StandardScaler().fit(df_num.values)
                    X_scaled = scaler_local.transform(df_num.values)
                else:
                    X_scaled = scaler.transform(df_num.values)

                X_seq = sliding_windows(X_scaled, timesteps=DEFAULT_LSTM_TIMESTEPS)
                proba = lstm_model.predict(X_seq).ravel()
                preds = (proba >= 0.5).astype(int)

                out = pd.DataFrame({"proba": proba, "pred": preds})
                st.dataframe(out.head(100))

                if y is not None:
                    y_true = y[DEFAULT_LSTM_TIMESTEPS:DEFAULT_LSTM_TIMESTEPS + len(preds)]
                    cm = confusion_matrix(y_true, preds)
                    st.pyplot(plot_confusion(cm, title="LSTM Confusion"))
                    fig_roc, auc_val = plot_roc(y_true, proba, label="LSTM")
                    st.pyplot(fig_roc)
                    st.success(f"ROC-AUC: {auc_val:.3f}")
            except Exception as e:
                st.error(f"LSTM inference failed: {e}")
