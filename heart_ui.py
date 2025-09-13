# import streamlit as st
# import joblib
# import pandas as pd


# model  = joblib.load("Logistic_heart.pkl")
# scaler = joblib.load("scaler.pkl")
# expected_columns = joblib.load("columns.pkl")




# st.title("Heart stroke prediction by harsh ❤️")
# st.markdown("Provide the following details")
# Age = st.slider("Age",18,100,40)
# Sex = st.selectbox("SEX",['M','F'])
# Chest_Pain = st.selectbox("Chest Pain Type",['ATA','NAP','TA','ASY'])
# Resting_bp = st.number_input("Resting Blood Pressure (mm Hg)",80,200,120)
# Cholesterol = st.number_input("Cholesterol (mg/dL)",100,600,200)
# Fasting_bs = st.selectbox("Fasting Blood Suger > 120 mg/dL",[0,1])
# Resting_ecg = st.selectbox("Resting ECG", ['Normal','ST','LVH'])
# Max_hr = st.slider("Max Heart Rate",60,220,150)
# Exercise_angina = st.selectbox("Exercise-Induced Angina",['Y','N'])
# Oldpeak = st.slider("Oldpeak (ST Depression)",0.0, 6.0, 1.0)
# St_slope = st.selectbox("ST Slope", ['Up','Flat','Down'])



# if st.button('Predict'):
#     raw_input = {
#         'Age': Age,
#         'RestingBP': Resting_bp,
#         'Cholesterol': Cholesterol,
#         'FastingBS' : Fasting_bs,
#         'MaxHR' : Max_hr,
#         'Oldpeak': Oldpeak,
#         'Sex_' + Sex: 1,
#         'ChestPainType_' + Chest_Pain: 1,
#         'RestingECG_' + Resting_ecg: 1,
#         'ExerciseAngina_' + Exercise_angina: 1,
#         'ST_Slope_' + St_slope: 1
#     }

#     input_df = pd.DataFrame([raw_input])

#     for col in expected_columns:
#         if col not in input_df.columns:
#             input_df[col] = 0
            
#     input_df = input_df[expected_columns]
    
#     scaled_input = scaler.transform(input_df)
#     prediction = model.predict(scaled_input)[0]
    
#     if prediction ==0:
#         st.error("⚠️ High Risk of Heart Disease")
#     else:
#         st.success("✅ Low Risk of Heart Disease")

import streamlit as st
import numpy as np
import pandas as pd
import joblib


model = joblib.load("KNN_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.set_page_config(
    page_title="Heart Stroke Prediction🫀",                              
    layout="centered",                         
    initial_sidebar_state="expanded"           
)
st.title("Heart Stroke Predictin By Harsh🫀")
st.markdown("⬇️Provide the following details to predict the likelihood of a heart stroke:-")


age = st.slider("Age",18,100,40)
sex = st.selectbox("SEX",['M','F'])
chest_pain = st.selectbox("Chest Pain Type",['ATA','NAP','TA','ASY'])
resting_bp = st.slider("Resting Blood Pressure (mm hg)",80,200,120)
cholesterol = st.slider("Cholesterol (mg/dl)",100,600,200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",[0,1])
resting_ecg = st.selectbox("Resting ECG", ["Normal","ST","LVH"])
Max_HR = st.slider("Max Heart Rate ",60,220,150)
excercise_angina = st.selectbox("Exercise Induced Angina",["Y","N"])
oldpeak = st.slider("Oldpeak (ST Depression)",0.0,6.0,1.0)
st_slope = st.selectbox("ST Slope",["Up","Flat","Down"])


if st.button("Predict"):
    raw_input = {
    'Age': age,
    'Sex_'+ sex:1,
    'ChestPainType_'+ chest_pain:1,
    'Cholesterol': cholesterol,
    'FastingBS': fasting_bs,
    'RestingBP': resting_bp,
    'RestingECG_'+ resting_ecg : 1,
    'MaxHR': Max_HR,
    'ExerciseAngina_'+ excercise_angina:1,
    'Oldpeak': oldpeak,
    'ST_Slope_'+ st_slope:1
    }


    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    
    if prediction == 0:
        st.error("⚠️high Risk Of Heart Dieases")
    else:
        st.success("❤️Low Risk Of Heart Dieases.")