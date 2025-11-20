import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Utility: Try to load using joblib first, then pickle
# -------------------------------
def robust_load(filepath):
    """Attempt to load a file using joblib, then pickle if needed."""
    try:
        import joblib
        return joblib.load(filepath)
    except Exception as e1:
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e2:
            raise RuntimeError(f"Failed to load {filepath} with joblib ({e1}) and pickle ({e2})")

# -------------------------------
# Load artifacts
# -------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = robust_load('model.pkl')
        scaler = robust_load('scaler.pkl')
        feature_columns = robust_load('feature_columns.pkl')
        st.success("✅ All artifacts (model, scaler, features) loaded successfully!")
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"❌ Failed to load required files: {e}")
        st.stop()

model, scaler, feature_columns = load_artifacts()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Diabetes Prediction - AshleyAI", page_icon="🩺")
st.title("🩺 AshleyAI: Diabetes Prediction App")
st.markdown("Enter patient metrics to predict diabetes risk using our trained AI model.")

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

# Convert to DataFrame **in the exact order used during training**
try:
    input_df = pd.DataFrame([{col: input_dict[col] for col in feature_columns}])
except KeyError as e:
    st.error(f"Missing feature in input: {e}. Ensure feature_columns.pkl matches the input fields.")
    st.stop()

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Diabetes Risk"):
    try:
        # Apply scaling
        input_scaled = scaler.transform(input_df)
        
        # Predict
        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        # Display result
        st.subheader("🎯 Prediction Result")
        if pred == 1:
            st.error("⚠️ **Prediction: Diabetic**")
        else:
            st.success("✅ **Prediction: Non-Diabetic**")

        st.subheader("📊 Confidence")
        st.write(f"Probability of **Non-Diabetic**: {proba[0]:.2%}")
        st.write(f"Probability of **Diabetic**: {proba[1]:.2%}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Ensure `scaler.pkl` and `feature_columns.pkl` match the training configuration.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("💡 Developed as part of the AI Mini Team Project. Not a medical diagnosis tool.")