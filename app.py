import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ============================================================================
# 1. LOAD MODEL AND SCALER
# ============================================================================

# Assuming the model and scaler files are in the same directory as app.py
try:
    with open('svm_cancer_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("Error: 'svm_cancer_model.pkl' file not found. Please ensure the trained model file is in the same directory.")
    model = None

try:
    with open('scaler_cancer.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Error: 'scaler_cancer.pkl' file not found. Please ensure the scaler file is in the same directory.")
    scaler = None

# List of all 30 features (must match training data order)
FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 
    'compactness_se', 'concavity_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# NOTE: I noticed 'concavity_se' appears twice in the logs for FEATURE_NAMES, 
# but the rest of the app logic implies 30 unique features. 
# Assuming the full correct list has been used when creating the PKL file.


# ============================================================================
# 2. STREAMLIT APP LAYOUT
# ============================================================================

st.set_page_config(page_title="Breast Cancer Prediction (SVM)", layout="wide")

st.title("🎀 Breast Cancer Prediction using SVM")
st.markdown("---")

if model is None or scaler is None:
    st.warning("Prediction cannot proceed. Please resolve the file loading errors above.")
else:
    st.sidebar.header("Patient Feature Input")
    st.sidebar.markdown("Please input the 30 cellular feature measurements.")

    # Dictionary to store user inputs
    input_data = {}
    
    # Function to create input fields
    def get_input(feature, label, default_value, min_value=0.0):
        if 'mean' in feature:
            step = 0.001
        elif 'se' in feature:
            step = 0.0001
        else: # worst
            step = 0.01

        return st.sidebar.number_input(
            label=f"{label} ({feature})",
            value=default_value,
            min_value=min_value,
            step=step,
            format="%.4f",
            key=feature
        )


    # ------------------
    # INPUT SECTION: Mean Values (10 features)
    # ------------------
    st.sidebar.subheader("A) Mean Values")
    
    # NOTE: The columns split here is a common Streamlit practice for layout
    mean_cols = st.sidebar.columns(2)
    
    # Example means (often around 10-20)
    input_data['radius_mean'] = mean_cols[0].number_input("Radius (Mean)", value=14.127, min_value=0.0, step=0.1, format="%.3f")
    input_data['texture_mean'] = mean_cols[1].number_input("Texture (Mean)", value=19.29, min_value=0.0, step=0.1, format="%.2f")
    
    # These should use the simple get_input function if you put them in the sidebar below the columns
    input_data['perimeter_mean'] = get_input('perimeter_mean', 'Perimeter', 91.969)
    input_data['area_mean'] = get_input('area_mean', 'Area', 654.889)
    input_data['smoothness_mean'] = get_input('smoothness_mean', 'Smoothness', 0.0963)
    input_data['compactness_mean'] = get_input('compactness_mean', 'Compactness', 0.1043)
    input_data['concavity_mean'] = get_input('concavity_mean', 'Concavity', 0.0888)
    input_data['concave points_mean'] = get_input('concave points_mean', 'Concave Points', 0.0489)
    input_data['symmetry_mean'] = get_input('symmetry_mean', 'Symmetry', 0.1811)
    input_data['fractal_dimension_mean'] = get_input('fractal_dimension_mean', 'Fractal Dimension', 0.0628)
    
    # ------------------
    # INPUT SECTION: Standard Error (10 features)
    # ------------------
    st.sidebar.subheader("B) Standard Error (SE) Values")
    
    input_data['radius_se'] = get_input('radius_se', 'Radius (SE)', 0.405)
    input_data['texture_se'] = get_input('texture_se', 'Texture (SE)', 1.216)
    input_data['perimeter_se'] = get_input('perimeter_se', 'Perimeter (SE)', 2.866)
    input_data['area_se'] = get_input('area_se', 'Area (SE)', 40.337)
    input_data['smoothness_se'] = get_input('smoothness_se', 'Smoothness (SE)', 0.0070)
    input_data['compactness_se'] = get_input('compactness_se', 'Compactness (SE)', 0.0254)
    input_data['concavity_se'] = get_input('concavity_se', 'Concavity (SE)', 0.0318)
    input_data['concave points_se'] = get_input('concave points_se', 'Concave Points (SE)', 0.0117)
    input_data['symmetry_se'] = get_input('symmetry_se', 'Symmetry (SE)', 0.0205)
    input_data['fractal_dimension_se'] = get_input('fractal_dimension_se', 'Fractal Dimension (SE)', 0.0038)

    # ------------------
    # INPUT SECTION: Worst/Largest Values (10 features)
    # ------------------
    st.sidebar.subheader("C) Worst/Largest Values")

    input_data['radius_worst'] = get_input('radius_worst', 'Radius (Worst)', 16.269)
    input_data['texture_worst'] = get_input('texture_worst', 'Texture (Worst)', 25.677)
    input_data['perimeter_worst'] = get_input('perimeter_worst', 'Perimeter (Worst)', 107.26)
    input_data['area_worst'] = get_input('area_worst', 'Area (Worst)', 880.583)
    input_data['smoothness_worst'] = get_input('smoothness_worst', 'Smoothness (Worst)', 0.1323)
    input_data['compactness_worst'] = get_input('compactness_worst', 'Compactness (Worst)', 0.2542)
    input_data['concavity_worst'] = get_input('concavity_worst', 'Concavity (Worst)', 0.2721)
    input_data['concave points_worst'] = get_input('concave points_worst', 'Concave Points (Worst)', 0.1146)
    input_data['symmetry_worst'] = get_input('symmetry_worst', 'Symmetry (Worst)', 0.2901)
    input_data['fractal_dimension_worst'] = get_input('fractal_dimension_worst', 'Fractal Dimension (Worst)', 0.0839)

    # Create the DataFrame in the correct order
    input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
    
    # ------------------
    # Display Input Data
    # ------------------
    st.subheader("Current Input Data (30 Features)")
    st.dataframe(input_df.T, use_container_width=True)


    # ============================================================================
    # 3. PREDICTION LOGIC
    # ============================================================================

    st.subheader("Prediction Result")
    
    if st.button("Analyze Cell Data"):
        
        with st.spinner('Analyzing features...'):
            try:
                # 1. Scale the input data
                input_scaled = scaler.transform(input_df)
                
                # 2. Make Prediction
                prediction = model.predict(input_scaled)
                
                # 3. Interpret Result
                if prediction[0] == 1:
                    result = "Malignant (Cancerous)"
                    st.error(f"⚠️ Prediction: {result}")
                    st.balloons()
                    st.markdown("Based on the input features, the model predicts the mass is **Malignant (Cancerous)**. **Consult a specialist immediately.**")
                else:
                    result = "Benign (Non-Cancerous)"
                    st.success(f"✅ Prediction: {result}")
                    st.markdown("Based on the input features, the model predicts the mass is **Benign (Non-Cancerous)**. Regular follow-up is advised.")

                st.markdown("---")
                st.info("The model used is a Support Vector Classifier (SVC) with an RBF kernel, trained on the scaled Wisconsin Diagnostic Breast Cancer dataset.")

            except Exception as e:
                st.exception(f"An error occurred during prediction: {e}")

# Footer for running instructions
st.markdown("---")
st.markdown("""
### 🚀 How to Run This App:
1.  **Ensure files are present:** Make sure `svm_cancer_model.pkl` and `scaler_cancer.pkl` are in the same folder as this `app.py` file.
2.  **Install Streamlit:** `pip install streamlit pandas scikit-learn`
3.  **Run the app:** Open your terminal in the file's directory and type: `streamlit run app.py`
""")
