# Industrial Predictive Maintenance and Explainable AI (XAI) Dashboard

A full-stack, end-to-end Machine Learning web application that predicts industrial equipment failures before they happen and uses Explainable AI (XAI) to diagnose the root engineering causes in real time. 

## 🌐 Live Demo
👉 **[https://jnxp4dmojme4dfzpd92qyp.streamlit.app]**

---

## 📌 Project Overview
Unscheduled machine downtime costs manufacturing industries billions annually. This project shifts maintenance strategies from reactive to proactive by deploying a robust **Random Forest Classifier** trained on industrial telemetry data. 

To bridge the gap between complex AI logic and field engineering, the system integrates **SHAP (SHapley Additive exPlanations)**. When an imminent failure is flag-checked, the dashboard dynamically isolates the specific physical attributes (such as mechanical stress, thermal limits, or tool wear) driving the risk, allowing maintenance crews to act with precision.

### Key Features
* **Real-Time Telemetry Simulation:** Interactive controls to adjust critical physical indicators including Rotational Speed, Torque, Tool Wear, and Operating Temperatures.
* **Predictive Diagnostics Engine:** Evaluates machine state instantaneously using a trained ensemble model.
* **Explainable AI (XAI) Root Cause Analysis:** Generates custom feature-impact visualizations for every unique machine state to explain *why* an anomaly is occurring.
* **Cloud Infrastructure:** Zero-downtime micro-deployment hosted globally for immediate field testing.

---

## 🛠️ System Architecture & Tech Stack

* **Language:** Python
* **Machine Learning Framework:** Scikit-Learn (Random Forest Ensemble)
* **Explainable AI:** SHAP Framework
* **Frontend UI / Hosting:** Streamlit & Streamlit Community Cloud
* **Data Pipelines & Visualizations:** Pandas, Matplotlib, Joblib

---

## 📊 Dataset Physical Indicators
The predictive engine analyzes the following physical parameters of the equipment:
* **Rotational Speed [rpm]:** Calculated from power grid frequencies with an expected range of 1000 to 3000 rpm.
* **Torque [Nm]:** Torsional forces applied to the cutting tool, normally centered around 40 Nm.
* **Tool Wear [min]:** Cumulative runtime tracking the degradation of the active machine element.
* **Air Temperature [K]:** Ambient structural room temperature surrounding the equipment.
* **Process Temperature [K]:** Internal thermal environment generated during active operational cycles.
