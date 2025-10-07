with st.expander("📘 About This Dashboard"):
    st.markdown("""
    ### ✈️ X-Plane Predictive Maintenance Dashboard
    This dashboard simulates **real-time engine health monitoring** for aircraft systems using live data from X-Plane.

    #### 🧩 Parameters:
    - **RPM**: Engine revolutions per minute — reflects power output.
    - **N1 / N2**: Turbine speeds (low & high pressure sections).
    - **EGT**: Exhaust Gas Temperature — a key early failure indicator.
    - **Oil Temp / Pressure**: Critical for lubrication and cooling.
    - **Fuel Pressure**: Indicates consistent flow; sudden drops can hint at pump or line faults.

    #### 🎯 Failure Probability Meter:
    - 🟢 0.00 – 0.40 → Stable (Engine healthy)
    - 🟡 0.41 – 0.70 → Low Risk (Potential warning signs)
    - 🔴 0.71 – 1.00 → High Risk (Immediate inspection advised)

    #### 💡 Powered by:
    - **XGBoost** (for static feature-based health scoring)
    - **LSTM Neural Network** (for temporal failure prediction)

    **Goal:** Predict failures before they happen — transforming maintenance from reactive to predictive.
    """)

It feels official — like a digital flight engineer’s quick reference guide.


---

⚙️ 2️⃣ Fault Localization (tell which part is failing)

Now this is very cool.
You’re talking about component-level diagnostics — i.e. not just “failure incoming” but “which subsystem is responsible”.

We can do this intelligently even without modifying the model yet:

✅ Option A (Heuristic-based fault explanation)
Define thresholds for parameters that usually indicate specific subsystem stress:

def identify_fault(row):
    faults = []
    if row["OILT1__deg"].values[0] > 110:
        faults.append("Oil System Overheating")
    if row["EGT_1__deg"].values[0] > 850:
        faults.append("Exhaust Temperature High")
    if row["FUEP1__psi"].values[0] < 20:
        faults.append("Fuel Pressure Drop")
    if row["N1__1_pcnt"].values[0] < 50 and row["rpm_1engin"].values[0] < 1500:
        faults.append("Combustion Instability")
    
    if not faults:
        return "No faults detected — system nominal."
    else:
        return ", ".join(faults)

Then just below your status meter:

fault_text = identify_fault(row)
st.markdown(f"**🧩 Detected Issue:** {fault_text}")

This gives users a contextual diagnosis, even before a full ML explainability model is built.


---

✅ Option B (Explainable AI – SHAP-based explanation) Later, we can plug in SHAP values for XGBoost to actually highlight which features contributed most to a failure probability spike — like a live “AI reasoning overlay”.

It’ll show:

> 🧠 “Model detected anomaly due to rising EGT and falling fuel pressure.”




---

So basically:

The Info Button helps the pilot understand the dashboard 🧾

The Fault Detector helps them understand the aircraft ⚙️



---

Would you like me to integrate both (the info expander + live fault detector logic) right into your existing dashboard code?
I’ll make sure it appears cleanly beside the meter — no clutter, just smart, pilot-style feedback.

