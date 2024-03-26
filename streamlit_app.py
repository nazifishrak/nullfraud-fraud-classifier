import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
# Load your trained model
# (Assuming your model is trained and saved as 'rf_classifier_balanced.pkl')
import joblib
rf_classifier_balanced = joblib.load('rf_classifier_balanced.pkl')

# Function to encode user inputs
def encode_inputs(user_inputs, label_encoders):
    encoded_inputs = user_inputs.copy()
    for col, le in label_encoders.items():
        # Convert date and time objects to strings
        if isinstance(user_inputs[col], (pd.Timestamp, pd.DatetimeIndex, pd.Period)):
            user_input = str(user_inputs[col].date())
        elif isinstance(user_inputs[col], (pd.Timedelta, pd.TimedeltaIndex)):
            user_input = str(user_inputs[col].time())
        else:
            user_input = user_inputs[col]
        
        # Handle unseen labels by assigning a default value (e.g., the mode of the training data)
        if user_input not in le.classes_:
            user_input = le.classes_[0]  # or some other default value
        
        encoded_inputs[col] = le.transform([user_input])[0]
    return encoded_inputs



# Function to make predictions
def predict_fraud(user_inputs_encoded):
    input_df = pd.DataFrame([user_inputs_encoded])
    prediction = rf_classifier_balanced.predict(input_df)
    probability = rf_classifier_balanced.predict_proba(input_df)[:, 1]
    print(prediction)
    print(probability)
    return prediction[0], probability[0]

# Streamlit interface
st.title('Fraud Detection System')

# Collect user inputs
user_inputs = {}
user_inputs['Card Identifier'] = st.text_input('Card Identifier')
user_inputs['Transaction Date'] = st.date_input('Transaction Date')
user_inputs['Transaction Time'] = st.time_input('Transaction Time')
user_inputs['Risk Assessment'] = st.number_input('Risk Assessment', min_value=0)
user_inputs['Payment Method'] = st.selectbox('Payment Method', ['Credit Card', 'Debit Card', 'Wire Transfer', 'Paypass - Contactless'])
user_inputs['Transaction Value'] = st.number_input('Transaction Value', min_value=0.0)
user_inputs['Merchant Location'] = st.text_input('Merchant Location')
user_inputs['Card Present Status'] = st.selectbox('Card Present Status', ['Present', 'Not Present'])
user_inputs['Chip Usage'] = st.selectbox('Chip Usage', ['Used', 'Not Used'])
user_inputs['Cross-border Transaction (Yes/No)'] = st.selectbox('Cross-border Transaction (Yes/No)', ['Yes', 'No'])
user_inputs['Acquiring Institution ID'] = st.text_input('Acquiring Institution ID')
user_inputs['Merchant Identifier'] = st.text_input('Merchant Identifier')
user_inputs['Merchant Category Code (MCC)'] = st.text_input('Merchant Category Code (MCC)')

# Load label encoders (Assuming they are saved as 'label_encoders.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Encode user inputs
encoded_inputs = encode_inputs(user_inputs, label_encoders)

# Make prediction
if st.button('Predict Fraud'):
    prediction, probability = predict_fraud(encoded_inputs)
    if prediction == 1:
        st.error(f'Fraud Detected with {probability*100:.2f}% probability')
    else:
        st.success(f'No Fraud Detected with {probability*100:.2f}% probability')
