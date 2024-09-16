import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model
# Ensure the model file 'lr.pkl' is in the same directory as this script
with open('lr.pkl') as file:
    model = joblib.load('lr.pkl')

# Set the title of the app
st.title("Diabetes Detection")



# Add a sidebar for manual input and file upload
st.sidebar.header("Input Options")

st.image("C:\\Users\\ACER\\Desktop\\diab\\download (1).jfif", caption="This is a local image", use_column_width=True)


# Sidebar selection for manual input or file upload
option = st.sidebar.radio("Choose input method:", ("Manual Entry", "Upload CSV"))

# Function to make predictions
def predict_diabetes(data):
    prediction = model.predict(data)
    probability = model.predict_proba(data)[0][1]  # Probability of having diabetes
    return prediction[0], probability

# Manual data entry section
if option == "Manual Entry":
    st.sidebar.subheader("Manual Data Entry")

    # Collecting diabetes-related data using number inputs
    pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=300, step=1)
    blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=200, step=1)
    skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
    insulin = st.sidebar.number_input("Insulin Level", min_value=0, max_value=900, step=1)
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0,step=0.1)
    diabetic_pedigree = st.sidebar.number_input("Diabetic Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
    age = st.sidebar.number_input("Age", min_value=16, max_value=120, step=1)
    
    # Display the entered data
    #st.write("## Entered Data:")
    #st.write(f"Pregnancies: {pregnancies}")
    #st.write(f"Glucose Level: {glucose}")
    #st.write(f"Blood Pressure: {blood_pressure}")
    #st.write(f"Skin Thickness: {skin_thickness}")
    #st.write(f"Insulin Level: {insulin}")
    #st.write(f"BMI: {bmi}")
    #st.write(f"Diabetic Pedigree Function: {diabetic_pedigree}")
    #st.write(f"Age: {age}")

    # Prepare data for prediction
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetic_pedigree, age]])

    # Prediction button
    if st.button("Predict"):
        prediction, probability = predict_diabetes(input_data)
        if prediction == 1:
            st.success(f"The model predicts that you have diabetes with a probability of {probability:.2f}.")
        else:
            st.success(f"The model predicts that you do not have diabetes with a probability of {1 - probability:.2f}.")

# CSV file upload section
elif option == "Upload CSV":
    st.sidebar.subheader("Upload a CSV File")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the CSV file
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check if the necessary columns are present
            expected_columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                                "Insulin", "BMI", "DiabeticPedigreeFunction", "Age"]
            if all(column in df.columns for column in expected_columns):
                # Display the CSV file data
                st.write("## Uploaded CSV Data:")
                st.write(df)
                
                # Prediction button for CSV data
                if st.button("Predict for CSV Data"):
                    predictions = model.predict(df)
                    probabilities = model.predict_proba(df)[:, 1]  # Probabilities of having diabetes
                    
                    df['Prediction'] = predictions
                    df['Probability'] = probabilities
                    st.write("## Prediction Results:")
                    st.write(df)
            else:
                st.error("The uploaded CSV does not contain the required columns.")
                
        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")
    else:
        st.write("Please upload a CSV file to display the data.")
