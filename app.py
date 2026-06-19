import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Setup the Page Layout
st.set_page_config(page_title="Equipment Diagnostic UI", layout="wide")
st.title("⚙️ Industrial Predictive Maintenance Dashboard")
st.markdown("Enter live telemetry to check the machine's mechanical health.")

# Hide the GitHub icon
hide_github_icon = """
            <style>
            /* Hide the GitHub link/icon */
            header a[href^="https://github.com"] { display: none !important; }
            /* Hide the Deploy button */
            [data-testid="stAppDeployButton"] { display: none !important; }
            .stDeployButton { display: none !important; }
            </style>
            """
st.markdown(hide_github_icon, unsafe_allow_html=True)

# 2. Load the AI Brain AND the SHAP Explainer
@st.cache_resource
def load_models():
    model = joblib.load('predictive_engine_model.pkl')
    features = joblib.load('model_features.pkl')
    explainer = joblib.load('shap_explainer.pkl')
    return model, features, explainer

rf_model, expected_features, explainer = load_models()

# 3. Create the User Interface (Sidebar for Inputs)
st.sidebar.header("📡 Live Sensor Inputs")

# Create sliders based on the exact physics of the dataset
rotational_speed = st.sidebar.slider("Rotational Speed [rpm]", 1000, 3000, 1500)
torque = st.sidebar.slider("Torque [Nm]", 10.0, 80.0, 40.0)
tool_wear = st.sidebar.slider("Tool Wear [min]", 0, 300, 50)
air_temp = st.sidebar.slider("Air Temperature [K]", 290.0, 310.0, 298.0)
process_temp = st.sidebar.slider("Process Temperature [K]", 300.0, 320.0, 310.0)

# Quality Type Dropdown
product_type = st.sidebar.selectbox("Product Quality Type", ["Low (L)", "Medium (M)", "High (H)"])
type_l = 1 if product_type == "Low (L)" else 0
type_m = 1 if product_type == "Medium (M)" else 0

# 4. Package the UI inputs into a format the AI understands
input_data = pd.DataFrame({
    'Air temperature [K]': [air_temp],
    'Process temperature [K]': [process_temp],
    'Rotational speed [rpm]': [rotational_speed],
    'Torque [Nm]': [torque],
    'Tool wear [min]': [tool_wear],
    'Type_L': [type_l],
    'Type_M': [type_m]
})

# Ensure the columns match exactly what the model expects
input_data = input_data[expected_features]

# 5. The Prediction & Root Cause Engine
if st.button("Run System Diagnostic"):
    with st.spinner("Analyzing telemetry & calculating root causes..."):
        # Get the prediction
        prediction = rf_model.predict(input_data)
        
        st.markdown("---")
        
        # Display Alert
        if prediction[0] == 1:
            st.error("🚨 **CRITICAL ALERT: Imminent Machine Failure Detected!**")
            st.warning("Action: Dispatch maintenance crew to inspect the unit immediately.")
        else:
            st.success("✅ **SYSTEM NORMAL: No mechanical anomalies detected.**")
            
        # Generate SHAP Root Cause Analysis
        st.subheader("📊 XAI Root Cause Analysis")
        st.markdown("*This chart explains the specific factors driving this prediction. Bars pushing to the right increase the risk of failure.*")
        
        # Calculate SHAP values for this specific input
        shap_values = explainer.shap_values(input_data)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Extract the failure class values (Class 1) and plot
        try:
            # Depending on scikit-learn/shap version, the array structure changes slightly
            shap_to_plot = shap_values[:, :, 1] 
        except TypeError:
            shap_to_plot = shap_values[1]
            
        shap.summary_plot(shap_to_plot, input_data, plot_type="bar", show=False)
        
        plt.xlabel("Impact Factor on Failure Risk")
        # Fix the cropped label margins
        plt.tight_layout()

        # Render plot in Streamlit
        st.pyplot(fig)