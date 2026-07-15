import streamlit as st 
from project import predict_survival

st.title("Titanic Survival Prediction")
st.write("Welcome to my Ai & ML Capstone Project")
import streamlit as st

st.title("Titanic Survival Prediction")

st.write("Enter passenger details below:")

pclass = st.selectbox("Passenger Class", [1, 2, 3])

sex = st.selectbox("Gender", ["Male", "Female"])

age = st.number_input("Age", min_value=0, max_value=100, value=25)

fare = st.number_input("Fare", min_value=0.0, value=32.0)


        
from project import predict_survival

if st.button("Predict"):
    result = predict_survival(pclass, sex, age, fare)

    if result == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")
        
        