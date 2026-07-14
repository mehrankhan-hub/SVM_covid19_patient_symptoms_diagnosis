import pandas as pd
import pickle as pkl
import streamlit as st
import numpy as np


df = pd.read_csv('covid19_patient_symptoms_diagnosis.csv')
#print(df)

with open('Covid19 Prediction.pkl','rb') as f:
    mdl = pkl.load(f)


st.title('🩺 COVID-19 Patient Symptom Diagnosis System')
st.image('C:\\Users\\toshiba\\PythonProject2\\Coronavirus-640w.webp',width =680)
st.subheader('Analyze patient symptoms to support accurate and timely COVID-19 diagnosis.')
gender = st.radio('Gender (1 = Male, 0 = Female)',(1,0))
age = st.number_input('Age',min_value = 0,max_value = 100)
fever = st.selectbox('Fever (1 = Yes, 0 = No)', (1,0))
dry_cough = st.selectbox('Dry Cough (1 = Yes, 0 = No)',(1,0))
sore_throat = st.selectbox('Sore Throat (1 = Yes, 0 = No)',(1,0))
fatigue = st.selectbox('Fatigue (1 = Yes, 0 = No)',(1,0))
headache = st.selectbox('Headache (1 = Yes, 0 = No)',(1,0))
shortness_of_breath = st.selectbox('Shortness of breath (1 = Yes, 0 = No)',(1,0))
loss_of_smell = st.selectbox('Loss of smell (1 = Yes, 0 = No)',(1,0))
loss_of_taste = st.selectbox('Loss of taste (1 = Yes, 0 = No)',(1,0))
oxygen_level = st.slider('Oxygen level',min_value = 70,max_value = 100,value = 85 )
body_temperature = st.slider('Body Temperature ',min_value = 25,max_value = 45,value = 35 )
comorbidity = st.radio('Comorbidity (1 = Diabetes, 0 = No)',(1,0))
travel_history = st.selectbox('Travel History (1 = Yes, 0 = No)',(1,0))
contact_with_patient  = st.selectbox('Contact with patient (1 = Yes, 0 =No)',(1,0))
chest_pain = st.selectbox('Chest pain (1 = Yes, 0 = No)',(1,0))

x_values = pd.DataFrame({
    'gender':[gender],'age':[age],'fever':[fever],'dry_cough':[dry_cough],
    'sore_throat':[sore_throat],'fatigue':[fatigue],'headache':[headache],
    'shortness_of_breath':[shortness_of_breath],'loss_of_smell':[loss_of_smell],
    'loss_of_taste':[loss_of_taste],'oxygen_level':[oxygen_level],'body_temperature':[body_temperature],
    'comorbidity':[comorbidity],'travel_history':[travel_history],'contact_with_patient':[contact_with_patient],
    'chest_pain':[chest_pain]
})

if st.button('Predict'):
    prediction = mdl.predict(x_values)
    if prediction ==0:
        st.success('COVID-19 Not Detected')
    else:
        st.error('Disease Detected')

st.header('Input Submitted for Processing')
st.table(x_values)


def footer():
    st.write("---")
    st.write("Heart Disease Prediction System")
    st.write("Developed by MEHRAN KHAN")
    st.write("© 2026 All Rights Reserved")
footer()



st.markdown("""
<style>

/* COVID-19 Medical Background */
.stApp {
    background:
        linear-gradient(
            rgba(230, 245, 250, 0.75),
            rgba(220, 250, 245, 0.75)
        ),
        url("https://images.unsplash.com/photo-1584036561566-baf8f5f1b144");

    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}


/* Main content glass card */
.block-container {
    background: rgba(255, 255, 255, 0.85);
    padding: 2.5rem;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 4px 20px rgba(0,0,0,0.15);
}


/* Text visibility */
.stApp h1,
.stApp h2,
.stApp h3 {
    color: #00695C !important;
    font-weight: 700;
}


.stApp p,
.stApp label,
.stApp span,
.stApp div {
    color: #263238 !important;
}


/* Input boxes */
.stTextInput input,
.stNumberInput input,
.stSelectbox div {
    background-color: rgba(255,255,255,0.9) !important;
    color: #263238 !important;
    border-radius: 10px;
}


/* Button styling */
.stButton button {
    background-color: #00897B !important;
    color: white !important;
    border-radius: 12px;
    padding: 10px 25px;
    font-weight: bold;
    border: none;
}


.stButton button:hover {
    background-color: #00695C !important;
}


/* Success / Warning messages */
.stSuccess {
    background-color: rgba(200, 240, 220, 0.9) !important;
}

.stWarning {
    background-color: rgba(255, 235, 180, 0.9) !important;
}


</style>
""", unsafe_allow_html=True)