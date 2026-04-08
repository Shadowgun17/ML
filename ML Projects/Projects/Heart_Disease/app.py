import streamlit as st 
import pandas as pd 
import joblib

model = joblib.load('Heart_Disease/LR.pkl')
scaler = joblib.load('Heart_Disease/scaler.pkl')
expected_columns = joblib.load('Heart_Disease/columns.pkl')

st.title('Heart Stroke Predictor')
st.markdown("Provide the Age")

age = st.slider('Age', 18,100,40)
sex = st.selectbox('Sex',['M','F'])
pain = st.selectbox('Chest Pain Type',['ATA','NAP','TA','ASY'])
restingBP = st.number_input('Resting Blood Pressure',80,200,120)
cholesterol = st.number_input("Cholesterol", 100,600,200)
fastingBS = st.selectbox('Fasting Blood Suger > 120 mg/dL',[0,1])
restingECG = st.selectbox('Resting ECG',['Normal','ST','LVH'])
maxHR = st.slider("Max Heart Rate",60,220,150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ['Y','N'])
oldpeak = st.slider('Oldpeak (st depression)', 0.0,6.0,1.0)
st_slope = st.selectbox('ST slope',['Up','Flat','Down'])


if st.button('Predict'):
    # 1. Create the initial DataFrame (Base Features)
    # Ensure these names match your training CSV columns EXACTLY
    raw_data = pd.DataFrame({
        'Age': [age],
        'RestingBP': [restingBP],
        'Cholesterol': [cholesterol],
        'FastingBS': [fastingBS],
        'MaxHR': [maxHR],
        'Oldpeak': [oldpeak],
        'Sex': [sex],
        'ChestPainType': [pain],
        'RestingECG': [restingECG],
        'ExerciseAngina': [exercise_angina],
        'ST_Slope': [st_slope]
    })

    input_df = pd.get_dummies(raw_data, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)

    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

   
    try:
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        if prediction == 1:
            st.error('High risk of Heart Disease')
        else:
            st.success('Low risk of Heart Disease')
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")

    