import streamlit as st
import pandas as pd
import pickle
import numpy as np

# -------------------------------
# Utility: Try to load using joblib first, then pickle
# -------------------------------
def robust_load(filepath):
    """Attempt to load a file using joblib, then pickle if needed."""
    try:
        import joblib
        return joblib.load(filepath)
    except ImportError:
        st.warning(f"joblib not found, trying pickle for {filepath}")
    except Exception as e1:
        st.warning(f"Loading {filepath} with joblib failed: {e1}. Trying pickle...")
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e2:
        raise RuntimeError(f"Failed to load {filepath} with joblib ({e1 if 'e1' in locals() else 'N/A'}) and pickle ({e2})")

# -------------------------------
# Load artifacts
# -------------------------------
@st.cache_resource
def load_model():
    return robust_load('model.pkl')

@st.cache_resource
def load_scaler():
    try:
        return robust_load('scaler.pkl')
    except RuntimeError:
        st.info("Scaler file 'scaler.pkl' not found or could not be loaded. Assuming no scaling was applied during training.")
        return None

@st.cache_resource
def load_feature_columns():
    try:
        return robust_load('feature_columns.pkl')
    except RuntimeError:
        st.warning("Feature columns file 'feature_columns.pkl' not found or could not be loaded. Using default column order from input.")
        return None

model = load_model()
scaler = load_scaler()
feature_columns = load_feature_columns()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Diabetes Prediction - AshleyAI", page_icon="🩺")
st.title("🩺 AshleyAI: Diabetes Risk Predictor")
st.markdown("Enter patient health metrics below to predict diabetes risk using our trained AI model.")

# -------------------------------
# Input form
# -------------------------------
st.subheader("🩻 Patient Health Metrics")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.slider("Glucose (mg/dL)", 0, 200, 120)
    blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 122, 70)
    skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)

with col2:
    insulin = st.number_input("Insulin (mu U/ml)", 0, 846, 80, step=1)
    bmi = st.slider("BMI", 0.0, 70.0, 25.0, step=0.1)
    diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.001)
    age = st.slider("Age (years)", 10, 100, 30)

# Build input dictionary
input_dict = {
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'BloodPressure': blood_pressure,
    'SkinThickness': skin_thickness,
    'Insulin': insulin,
    'BMI': bmi,
    'DiabetesPedigreeFunction': diabetes_pedigree,
    'Age': age
}

# Create DataFrame
input_df = pd.DataFrame([input_dict])

# Reorder DataFrame if feature_columns is available
if feature_columns is not None:
    try:
        # Ensure the order matches training
        input_df = input_df.reindex(columns=feature_columns)
    except KeyError as e:
        st.error(f"Feature mismatch: {e}. Check if 'feature_columns.pkl' matches input fields.")
        st.stop()

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Diabetes Risk"):
    try:
        # Prepare data for prediction
        X_to_predict = input_df.values  # Get the values as numpy array

        # Apply scaling if scaler is loaded
        if scaler is not None:
            X_scaled = scaler.transform(X_to_predict)
            prediction = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]
        else:
            # Use raw values if no scaler
            prediction = model.predict(X_to_predict)[0]
            proba = model.predict_proba(X_to_predict)[0]

        # Display result
        st.subheader("✅ Prediction Result")
        if prediction == 1:
            st.error("⚠️ **Prediction: Likely HAS Diabetes**")
        else:
            st.success("✅ **Prediction: Unlikely to Have Diabetes**")

        st.subheader("📊 Confidence Level")
        st.write(f"Probability of **No Diabetes**: {proba[0]:.2%}")
        st.write(f"Probability of **Diabetes**: {proba[1]:.2%}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.info("Ensure `model.pkl` (and `scaler.pkl`, `feature_columns.pkl` if used) are compatible with the input features.")

# -------------------------------
# Info Footer
# -------------------------------
st.markdown("---")
st.caption("💡 Developed as part of the AI Mini Team Project. Not a medical diagnosis tool.")
