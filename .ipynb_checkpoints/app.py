import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Load the model
diabetes = joblib.load('lr.pkl')

st.title("DIABETES PREDICTION")

st.sidebar.title("Input Options")
input_method = st.sidebar.selectbox("How would you like to input your data?", ("Manual Input", "Upload CSV"))

def make_predictions(input_data):
    prediction = diabetes.predict(input_data)
    return "DIABETIC" if prediction[0] == 1 else "NOT DIABETIC"
if input_method == "Manual Input":
    st.write("Enter the patient's data to predict the likelihood of diabetes.")
# Input features
Pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
Glucose = st.number_input("Glucose Level", min_value=0, max_value=200, step=1)
BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, step=1)
SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0, step=1)
Insulin = st.number_input("Insulin (mu U/ml)", min_value=0, step=1)
BMI = st.number_input("BMI (kg/m^2)", min_value=0.0, step=0.1)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
Age = st.number_input("Age", min_value=17, step=1)

# Prediction button
if st.button("PREDICT"):
    # Prepare the input data for prediction
    input_data = np.array([[Pregnancies, Glucose, BloodPressure , SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    result = make_predictions(input_data)
    st.write(f"The model predicts that the patient id **{result}**.")
elif input_method == "Upload CSV":
    st.write("Upload a CSV file with the same columns: 'Pregnancies', 'Glucose', 'BloodPressure' , 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'")

    upload_file = st.file_uploader("Choose a CSV file", type="csv")

    if upload_file is not None:

        data = pd.read_csv(upload_file)

        required_columns = ['Pregnancies', 'Glucose', 'BloodPressure' , 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        if all(col in data.columns for col in required_columns):
            predictions = diabetes.predict(data[required_columns])
            data['Prediction'] = predictions
            data['Prediction'] = pdata['Prediction'].apply(lambda x: "Diabetic" if x == 1 else "Not Diabetic")

            st.write("Prediction:")
            st.write(data)

            csv = data.to_csv(index=False)
            st.download_button("Download P redictions",csv, "predictions.csv","text/csv")

        else:
            st.write("The upload CSV does not have the required columns.")

