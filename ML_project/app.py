from pycaret.regression import load_model, predict_model
from logzero import logger
import pickle
import numpy as np
import streamlit as st
import pandas as pd

# model =load_model('insurance_prediction')
# dataset=pd.read_excel('insurance_charges_dataset.xlsx')

st.set_page_config(page_title = "Insurance Charges Prediction")

# @st.cache(allow_output_mutation=True)
def get_model():
    return load_model("insurance_prediction")

def predict(model, data):
    prediction = predict_model(model, data=data)
    return prediction['prediction_label'][0]

model  = get_model()

st.title("Insurance Charges Prediction")

form = st.form('charges')
age = form.number_input('Age', min_value=1, max_value=100, value=25)
sex = form.radio('Sex', ['Male', 'Female'])
bmi = form.number_input('BMI', min_value=10.0, max_value=50.0, value=20.0)
children = form.slider('Children', min_value=0, max_value=10, value=0)
region_list = ['Southwest', 'Northwest', 'Northeast', 'Southeast']
region = form.selectbox('Region', region_list)
# insurance_charges = st.text_area("charges")
if form.checkbox('Smoker'):
    smoker = 'yes'
else:
    smoker = 'no'
    
predict_button = form.form_submit_button('Predict')

input_dict = {
    'age':age,
    'sex':sex.lower(),
    'bmi':bmi,
    'children': children,
    'smoker': smoker,
    'region': region.lower()
}

input_df = pd.DataFrame([input_dict])

if predict_button:
    output = predict(model, input_df)
    st.success("The predicted charges are {: .2f}".format(output))