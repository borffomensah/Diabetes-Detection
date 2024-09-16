import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Load the model
diabetes = joblib.load('lr.pkl')

st.title("DIABETES PREDICTION")


# Display an image from a URL
#st.image("https://www.google.com/imgres?q=diabetes&imgurl=https%3A%2F%2Fcuidadores.unir.net%2Fimages%2F7SintomasDiabetes.jpg&imgrefurl=https%3A%2F%2Fcuidadores.unir.net%2Finformacion%2Factualidad%2F609-7-sintomas-de-la-diabetes&docid=xWKhngmPQr8uzM&tbnid=-rj-TKhwXj3zuM&vet=1&w=870&h=435&hcb=2&ved=2ahUKEwiPuvCUlqSIAxVf0wIHHRXlAJIQM3oECFwQAA", caption="This is an image from a URL", use_column_width=True)


# Display a local image using doubled backslashes
st.image("C:\\Users\\ACER\\Desktop\\diab\\download (1).jfif", caption="This is a local image", use_column_width=True)



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
    
    # Make prediction
    prediction = diabetes.predict(input_data)
    prediction_prob = diabetes.predict_proba(input_data)

    # Display prediction result
    if prediction[0] == 1:
        st.subheader("DIABETES DETECTED")
    else:
        st.subheader("NO DIABETES DETECTED")

    # Display prediction probability
    st.write(f"Prediction Probability: {prediction_prob[0][1]*100:.2f}% chance of having diabetes")

    # Display feature importance
     #feature_importance = logreg.coef_[0]
     #features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree Function', 'Age']
    
     #importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
     #importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
     #st.subheader("Feature Importance")
     #st.bar_chart(importance_df.set_index('Feature'))