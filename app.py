import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open("best_model.pkl", "rb") as file:
    gb_reg = pickle.load(file)

def app():
    st.title('Calorie Burnt Prediction')
    st.write('Enter the values for the independent variables and click predict to see the calories burnt value')

    Duration = st.number_input("Duration(in minutes)")
    heart_rate = st.number_input("Heart Rate(in beats per minutes)")
    body_temp = st.number_input("Body Temperature(in degree celsius)")

    prediction = gb_reg.predict([[Duration,heart_rate,body_temp]])

    st.write("Predicted Calories Burnt:",prediction[0])


if __name__ == '__main__':
    app()