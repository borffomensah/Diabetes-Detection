import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

# Load the pre-trained model
with open('lr.pkl', 'rb') as file:
    model = joblib.load(file)


# Rest of the code for Streamlit app
st.title("Diabetes Detection: Manual Input and CSV Upload")

# Add a sidebar for manual input and file upload
st.sidebar.header("Input Options")

# Sidebar selection for manual input or file upload
option = st.sidebar.radio("Choose input method:", ("Manual Entry", "Upload CSV"))

# Function to make predictions
def predict_diabetes(data):
    prediction = model.predict(data)
    probability = model.predict_proba(data)[0][1]  # Probability of having diabetes
    return prediction[0], probability

# Function to generate the report
def generate_report(name, sex, data, prediction, probability):
    doc = Document()
    doc.add_heading("Diabetes Prediction Report", 0)
    
    # Adding patient information
    doc.add_paragraph(f"Name: {name}")
    doc.add_paragraph(f"Sex: {sex}")
    
    # Adding input data
    doc.add_heading("Entered Data:", level=1)
    doc.add_paragraph(f"Pregnancies: {data[0][0]}")
    doc.add_paragraph(f"Glucose Level: {data[0][1]}")
    doc.add_paragraph(f"Blood Pressure: {data[0][2]}")
    doc.add_paragraph(f"Skin Thickness: {data[0][3]}")
    doc.add_paragraph(f"Insulin Level: {data[0][4]}")
    doc.add_paragraph(f"BMI: {data[0][5]}")
    doc.add_paragraph(f"Diabetic Pedigree Function: {data[0][6]}")
    doc.add_paragraph(f"Age: {data[0][7]}")
    
    # Adding prediction results
    doc.add_heading("Prediction Results:", level=1)
    if prediction == 1:
        doc.add_paragraph(f"The model predicts that you have diabetes with a probability of {probability:.2f}.")
    else:
        doc.add_paragraph(f"The model predicts that you do not have diabetes with a probability of {1 - probability:.2f}.")
    
    # Save the document in-memory
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# Manual data entry section
if option == "Manual Entry":
    st.sidebar.subheader("Manual Data Entry")

    # Collecting diabetes-related data using number inputs
    pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
    glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=300, value=100, step=1)
    blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=200, value=80, step=1)
    skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)
    insulin = st.sidebar.number_input("Insulin Level", min_value=0, max_value=900, value=150, step=1)
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    diabetic_pedigree = st.sidebar.number_input("Diabetic Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=25, step=1)
    
    # Display the entered data
    st.write("## Entered Data:")
    st.write(f"Pregnancies: {pregnancies}")
    st.write(f"Glucose Level: {glucose}")
    st.write(f"Blood Pressure: {blood_pressure}")
    st.write(f"Skin Thickness: {skin_thickness}")
    st.write(f"Insulin Level: {insulin}")
    st.write(f"BMI: {bmi}")
    st.write(f"Diabetic Pedigree Function: {diabetic_pedigree}")
    st.write(f"Age: {age}")

    # Prepare data for prediction
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetic_pedigree, age]])

    # User information for the report
    name = st.text_input("Enter your name:")
    sex = st.selectbox("Select your sex:", ["Male", "Female", "Other"])

    # Prediction button
    if st.button("Predict"):
        if name:
            prediction, probability = predict_diabetes(input_data)
            if prediction == 1:
                st.success(f"The model predicts that you have diabetes with a probability of {probability:.2f}.")
            else:
                st.success(f"The model predicts that you do not have diabetes with a probability of {1 - probability:.2f}.")
            
            # Generate report
            report = generate_report(name, sex, input_data, prediction, probability)
            st.download_button(
                label="Download Report",
                data=report,
                file_name="Diabetes_Prediction_Report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        else:
            st.error("Please enter your name before proceeding.")
        
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
