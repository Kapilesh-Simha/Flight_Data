# ✈️ X-Plane Predictive Maintenance Dashboard

**Unified Real-Time + Simulation Dashboard** for aircraft engine health monitoring — powered by **XGBoost** and **LSTM** models.  
This Streamlit-based app predicts potential engine failures using telemetry from **X-Plane 11** or a built-in **interactive simulator**.

---

## 🚀 Features

### 🧠 Machine Learning
- **XGBoost Model** → Feature-based real-time failure classification.  
- **LSTM Model** → Sequential time-series analysis for anomaly detection.  
- **Unified Prediction** → Combines both models to yield a robust *failure probability score*.

### 📊 Visualization
- Real-time updating **failure probability gauge**.
- Dual **probability trend graphs** (XGBoost + LSTM).
- Detailed **telemetry summary** and engine insights.
- Adjustable **risk thresholds** (Green, Yellow, Red).

### 🕹️ Telemetry Simulator (Manual Mode)
- Full control over **34 flight and engine parameters**.
- Throttle-based auto mapping for realistic behavior.
- Fault simulation modes:
  - Engine Overheat 🔥
  - Fuel Restriction ⛽
  - Electronic Fuel Pump Failure ⚡
- Real-time updates to gauge and trend graphs.

### 🧾 Logging & Monitoring
- Automatic live logging of predictions into `live_log.csv`.
- Real-time append with timestamps and zone classification.
- Downloadable CSV via Streamlit UI.

### ⚙️ CI/CD (Jenkins Integration)
- Automated pipeline using `Jenkinsfile`:
  1. Virtual environment creation
  2. Dependency installation
  3. Model validation (XGBoost + LSTM)
  4. Streamlit dashboard launch on build completion

---

## 🧩 Project Structure

xplane_predictive_project/ │ ├── data/ │   ├── processed/ │   │   └── xplane_features.csv │   ├── live_log.csv                # Auto-created log file │ ├── models/ │   ├── xplane_xgboost.pkl │   ├── xplane_lstm.h5 │   └── lstm_scaler.pkl │ ├── src/ │   └── live_app.py                 # Main Streamlit app │ ├── Jenkinsfile                     # CI/CD pipeline for Jenkins ├── requirements.txt └── README.md

---

## 🛠️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/xplane_predictive_project.git
cd xplane_predictive_project

2️⃣ Create a Virtual Environment

python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate # On macOS/Linux

3️⃣ Install Dependencies

pip install --upgrade pip
pip install -r requirements.txt

4️⃣ Launch the Dashboard

streamlit run src/live_app.py

Then open your browser and visit:
👉 http://localhost:8501


---

💻 Jenkins Setup (CI/CD Pipeline)

1. Open Jenkins and create a new Pipeline project.


2. In Pipeline script from SCM, link this repository.


3. Build — Jenkins will:

Create a virtual environment.

Install dependencies.

Validate both models.

Launch the Streamlit app automatically (no manual localhost click needed).




> 💡 The dashboard becomes accessible locally at
http://localhost:8501




---

⚙️ Configuration Paths

Edit these paths in live_app.py if your directory structure differs:

PROJECT_DIR = "C:/Users/<user>/Desktop/xplane_predictive_project"
XGB_MODEL_PATH = f"{PROJECT_DIR}/models/xplane_xgboost.pkl"
LSTM_MODEL_PATH = f"{PROJECT_DIR}/models/xplane_lstm.h5"
SCALER_PATH = f"{PROJECT_DIR}/models/lstm_scaler.pkl"
DATA_PATH = f"{PROJECT_DIR}/data/processed/xplane_features.csv"
LOG_OUT_PATH = f"{PROJECT_DIR}/data/live_log.csv"


---

🧪 Modes

Mode	Description

📡 Real-Time Streaming	Simulates live data flow from X-Plane 11 telemetry.
🕹️ Simulator Telemetry Mode	Manually adjust parameters via sliders to test model behavior.
📊 Batch Analysis	Upload and analyze stored flight CSV data.



---

📈 Logging & Output

Logs are automatically appended to:

data/live_log.csv

Each log entry includes:

Timestamp	XGB Prob	LSTM Prob	Combined	Zone

2025-10-10T12:05:33+05:30	0.66	0.02	0.68	🔴 HIGH RISK



---

🧰 Troubleshooting

❌ “elementType 'numberInput' is not a valid arrowAddRows target”

This happens when Streamlit reuses widget IDs between charts and sliders.
✅ Fixed by isolating chart placeholders using init_safe_placeholders() (already in latest code).

❌ “Localhost refused to connect”

Ensure Streamlit is not running twice.
Stop existing processes:

taskkill /F /IM python.exe

Then restart Jenkins or Streamlit manually.

❌ “Scaling issue: X has 20 features but StandardScaler expects 34”

Ensure your telemetry CSV contains all 34 features in FULL_FEATURE_LIST.


---

🧠 Technologies Used

Category	Technology

Frontend	Streamlit, Plotly
ML Models	XGBoost, TensorFlow (LSTM)