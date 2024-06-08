import streamlit as st
import math
import pickle
import numpy as np

with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)


# Define the function for making predictions
def predict(features):
    features = np.array(features).reshape(1, -1)  # Reshape features to 2D array
    prediction = model.predict(features)
    return prediction[0]
# ['Pclass', 'Age', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q',
#        'Embarked_S', 'Family']


#streamlit app

html_temp = """<div style ="background-color:tomato; padding:10px;">
        <h2 style ="color:white; text-align:center;"> Titanic Survival Prediction Model.  </h2>
    """
st.markdown(html_temp, unsafe_allow_html=True)
st.write("Input the values to predict")

#collect input
col1, col2, col3 = st.columns(3)
with col1:
    pclass_name = st.selectbox("Class", ("Business Class", "First Class", "Economy"))
with col2:
    gender = st.selectbox("Gender", ["male", "female"], index=0)
with col3:
    age = st.number_input("Age", min_value=0, max_value=100, value=30)

col4, col5 = st.columns(2)
with col4:
    embarked = st.selectbox("Port of Embarkation", ["Cherbourg", "Queenstown", "Southampton"], index=0)
with col5:
    family = st.number_input("Family Members", min_value=0, max_value=10, value=0)

# Map passenger class name to numerical value
if pclass_name == "Business Class":
    pclass = 1
elif pclass_name == "First Class":
    pclass = 2
else:
    pclass = 3

# Encode categorical variables
sex_male = 1 if gender == "male" else 0
sex_female = 1 if gender == "female" else 0

embarked_C = 1 if embarked == "Cherbourg" else 0
embarked_Q = 1 if embarked == "Queenstown" else 0
embarked_S = 1 if embarked == "Southampton" else 0

# Arrange the features as per the model requirement
features = [pclass, age, sex_female, sex_male, embarked_C, embarked_Q, embarked_S, family]

# Make prediction
if st.button("Predict"):
    prediction = predict(features)
    if prediction == 1:

         st.header("The passenger would have survived.")
    else:
        st.header("The passenger would not have survived.")
