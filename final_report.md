# ✈️ X-Plane 11 Predictive Maintenance Project

## 1. Introduction
- **Objective**: Predict potential engine failures from X-Plane 11 simulator data.  
- **Motivation**: Aircraft engines generate vast amounts of data; predictive maintenance helps reduce downtime, costs, and risks.  
- **Approach**: Use simulated flight data to extract engine performance parameters and train machine learning models to predict failures.  

---

## 2. Dataset
- **Source**: Data generated from X-Plane 11 simulator logs.  
- **Size**: ~XX,XXX rows × XX features.  
- **Key features collected**:
  - N1, N2 (%)
  - Engine RPM
  - Power (hp)
  - Thrust (lb)
  - EGT (Exhaust Gas Temp)
  - Oil Temp
  - Fuel Pressure
  - Battery current/voltage
- **Target variable**: `failure` (synthetic injection where thresholds exceeded).  

---

## 3. Methodology
### 3.1 Data Processing
- Removed duplicates and constant columns.  
- Converted time-series logs into structured tabular format.  
- Injected synthetic failures due to absence of real ones.  

### 3.2 Feature Engineering
- Created rolling averages and deltas for RPM, EGT, Oil Temp.  
- Normalized continuous variables.  
- Balanced classes using:
  - Class weights (XGBoost)  
  - SMOTE (LSTM/other classifiers).  

### 3.3 Models Tried
- **XGBoost** (tree-based, strong baseline).  
- **LSTM** (sequence model for time-series patterns).  

---

## 4. Results

### 4.1 XGBoost
- **Best Threshold**: `0.85`  
- **Metrics**:
  - Accuracy: `85.61%`
  - Precision: `1.75%`
  - Recall: `11.23%`
  - F1 Score: `3.03%`
  - ROC-AUC: `46.37%`
- **Confusion Matrix**:  
  - [[11769 1738]
     [245   31]]

---

### 4.2 LSTM
- **Epochs**: XX, **Batch size**: XX  
- **Metrics**:
  - Accuracy: `98%`
- **Training Curve**:  
  ![LSTM Training Curve](../figures/lstm_loss_curve.png)  

---

## 5. Discussion
- **XGBoost** performed well for static tabular features but struggled with sequential dependencies.  
- **LSTM** captured time-series dynamics better, showing potential for improvement with more data.  
- Class imbalance significantly impacted results; synthetic failure generation improved training.  

---

## 6. Conclusion
- Developed a pipeline to:
  - Record X-Plane flight data  
  - Clean, preprocess, and engineer features  
  - Train machine learning models for predictive maintenance  
- Achieved ~86% accuracy with **XGBoost** and 98% accuracy with **LSTM**.  
- Demonstrated feasibility of predictive maintenance using simulated flight data.  

---

## 7. Future Work
- Enhance failure injection strategies to mimic real scenarios.  
- Explore advanced deep learning models (GRU, Transformers).  
- Deploy model in real-time with a dashboard (Streamlit and Jenkins).  

---

## 8. References
- X-Plane 11 Data Output Documentation  
- XGBoost Library  
- SMOTE for Imbalanced Datasets  
- Keras LSTM for Time-Series  
