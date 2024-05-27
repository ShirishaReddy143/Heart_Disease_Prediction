import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("""
# Heart disease Prediction App

This app predicts if a patient has a heart disease.""")

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
def user_input_features():
    age = st.sidebar.number_input('Enter your age: ')

    sex  = st.sidebar.selectbox('Sex',('male','female'))
    cp = st.sidebar.selectbox('Chest pain type',(0,1,2,3))
    tres = st.sidebar.number_input('Resting blood pressure: ')
    chol = st.sidebar.number_input('Serum cholestoral in mg/dl: ')
    fbs = st.sidebar.selectbox('Fasting blood sugar',(0,1))
    res = st.sidebar.number_input('Resting electrocardiographic results: ')
    tha = st.sidebar.number_input('Maximum heart rate achieved: ')
    exa = st.sidebar.selectbox('Exercise induced angina: ',(0,1))
    old = st.sidebar.number_input('Oldpeak ')
    slope = st.sidebar.number_input('The slope of the peak exercise ST segmen: ')
    ca = st.sidebar.selectbox('Number of major vessels',(0,1,2,3))
    thal = st.sidebar.selectbox('Thal',(0,1,2))

    data = {'age': age,
            'sex': sex, 
            'cp': cp,
            'trestbps':tres,
            'chol': chol,
            'fbs': fbs,
            'restecg': res,
            'thalach':tha,
            'exang':exa,
            'oldpeak':old,
            'slope':slope,
            'ca':ca,
            'thal':thal
           }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Reads in saved classification model
load_clf = pickle.load(open('model786.pkl', 'rb'))

# Apply one-hot encoding to user input
input_df['sex'] = input_df['sex'].map({'male': 1, 'female': 0})
input_df = pd.get_dummies(input_df, columns=['cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# Ensure input DataFrame matches the feature names used during training
expected_features = load_clf.feature_names_in_
input_df = input_df.reindex(columns=expected_features, fill_value=0)

# Predict function
def predict():
    prediction = load_clf.predict(input_df)
    prediction_proba = load_clf.predict_proba(input_df)*100
    if prediction == [None]:
        prediction = 'Please enter the data'
    elif prediction == [0]:
        prediction = 'U have suffering some Heart disease'
    else:
        prediction = 'At Present U r GOOD keep itup'
    return prediction, prediction_proba

if st.button('Predict'):
    prediction, prediction_proba = predict()
    st.subheader('Prediction')
    st.write(prediction)

    st.subheader('Probability of Prediction of Heart Disease')
    st.write(prediction_proba)
