import streamlit as st
import pandas as pd
import joblib

# Load model files
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

st.title("Heart Disease Prediction")

st.write("Enter the patient details below:")

# ----------- Inputs -----------

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ----------- Prediction -----------

if st.button("Predict"):

    try:
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'MaxHR': max_hr,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        input_df = pd.DataFrame([raw_input])

        # Add missing columns
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[expected_columns]

        scaled_input = scaler.transform(input_df)

        prediction = model.predict(scaled_input)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(scaled_input)[0][1] * 100
        else:
            probability = None

        st.subheader("Result")

        if prediction == 1:
            st.error("High Risk of Heart Disease")
        else:
            st.success("Low Risk of Heart Disease")

        if probability is not None:
            st.write(f"Risk Probability: {probability:.2f}%")

    except Exception as e:
        st.error(f"Error: {e}")