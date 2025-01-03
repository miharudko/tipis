import pandas as pd
import joblib

import streamlit as st
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(current_dir, 'credit_default_model.pkl')
model = joblib.load(model_path)

metadata_path = os.path.join(current_dir, 'model_metadata.pkl')
metadata = joblib.load(metadata_path)

categorical_features = metadata.get('categorical_features', ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose'])

feature_labels = {
    'Age': 'Age of the applicant',
    'Sex': 'Sex (0: Female, 1: Male)',
    'Job': 'Type of job (integer)',
    'Housing': 'Housing status (0: Rent, 1: Own, 2: Free)',
    'Saving accounts': 'Saving accounts status',
    'Checking account': 'Checking account status',
    'Credit amount': 'Requested credit amount',
    'Duration': 'Loan duration (months)',
    'Purpose': 'Purpose of the loan'
}

def get_user_input():
    user_data = {}
    for feature in metadata['columns']:
        if feature in categorical_features:
            label = feature_labels[feature]
            options = metadata['label_encoders'][feature].classes_
            selected_option = st.selectbox(label, options)
            user_data[feature] = metadata['label_encoders'][feature].transform([selected_option])[0]
        else:
            label = feature_labels[feature]
            value = st.number_input(label, value=0.0)
            user_data[feature] = value

    input_df = pd.DataFrame([user_data], columns=metadata['columns'])
    return input_df

st.title("Credit Default Predictor")

user_input = get_user_input()

st.write("Entered Values:")
st.write(user_input)

if st.button("Predict"):
    # Predict probability
    probability = model.predict_proba(user_input)[0][1]  # Probability of default
    st.subheader(f"Probability of Default: {probability:.2f}")
    if probability > 0.5:
        st.warning("High risk of default")
    else:
        st.success("Low risk of default")
