import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
rf_classifier_balanced = joblib.load('rf_classifier_balanced.pkl')


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
        

        if user_input not in le.classes_:
            user_input = le.classes_[0]  # or some other default value
        
        encoded_inputs[col] = le.transform([user_input])[0]
    return encoded_inputs



def predict_fraud(user_inputs_encoded):
    input_df = pd.DataFrame([user_inputs_encoded])
    prediction = rf_classifier_balanced.predict(input_df)
    probability = rf_classifier_balanced.predict_proba(input_df)[:, 1]
    print(prediction)
    print(probability)
    return prediction[0], probability[0]


st.title('Fraud Detection System')

# Collect user inputs
user_inputs = {}
user_inputs['Card Identifier'] = st.text_input('Card Identifier')
user_inputs['Transaction Date'] = st.date_input('Transaction Date')
user_inputs['Transaction Time'] = st.time_input('Transaction Time')
user_inputs['Risk Assessment'] = st.number_input('Risk Assessment', min_value=0)
user_inputs['Payment Method'] = st.selectbox('Payment Method',[
    'Chip',
    'eCommerce',
    'Magnetic Stripe',
    'Mobile Wallet',
    'Online',
    'Paypass - Contactless',
    'PayPass - Wallet',
    'Phone',
    'Postal',
    'Subscription',
    'Tap-to-Pay',
    'Unknown'
])
user_inputs['Transaction Value'] = st.number_input('Transaction Value', min_value=0.0)
user_inputs['Merchant Location'] = st.selectbox('Merchant Location',
[
    'ABW', 'AGO', 'ALB', 'AND', 'ARE', 'ARG', 'ARM', 'ATG', 'AUS', 'AUT', 'AZE',
    'BEL', 'BGR', 'BHR', 'BHS', 'BIH', 'BLM', 'BLR', 'BLZ', 'BMU', 'BOL', 'BRA',
    'BRB', 'CAN', 'CHE', 'CHL', 'CHN', 'COK', 'COL', 'CRI', 'CUW', 'CYM', 'CYP',
    'CZE', 'DEU', 'DNK', 'DOM', 'ECU', 'EGY', 'ESP', 'EST', 'ETH', 'FIN', 'FJI',
    'FRA', 'FSM', 'GBR', 'GEO', 'GGY', 'GHA', 'GIB', 'GLP', 'GRC', 'GTM', 'GUY',
    'HKG', 'HND', 'HRV', 'HUN', 'IDN', 'IMN', 'IND', 'IRL', 'IRQ', 'ISL', 'ISR',
    'ITA', 'JAM', 'JOR', 'JPN', 'KAZ', 'KEN', 'KGZ', 'KHM', 'KNA', 'KOR', 'LBN',
    'LKA', 'LTU', 'LUX', 'LVA', 'MAR', 'MCO', 'MDV', 'MEX', 'MKD', 'MLT', 'MNE',
    'MTQ', 'MUS', 'MYS', 'NGA', 'NLD', 'NOR', 'NZL', 'OMN', 'PAK', 'PAN', 'PER',
    'PHL', 'POL', 'PRI', 'PRT', 'PRY', 'PYF', 'QAT', 'QZZ', 'ROM', 'RWA', 'SAU',
    'SEN', 'SGP', 'SLV', 'SRB', 'SVK', 'SVN', 'SWE', 'SXM', 'SYC', 'TCA', 'THA',
    'TTO', 'TUR', 'TWN', 'TZA', 'UGA', 'UKR', 'URY', 'USA', 'UZB', 'VAT', 'VEN',
    'VGB', 'VIR', 'VNM', 'ZAF', 'ZMB'
])
user_inputs['Card Present Status'] = st.selectbox('Card Present Status', ['CNP', 'CP'])
user_inputs['Chip Usage'] = st.selectbox('Chip Usage', ['Yes', 'No'])
user_inputs['Cross-border Transaction (Yes/No)'] = st.selectbox('Cross-border Transaction (Yes/No)', ['Yes', 'No'])
user_inputs['Acquiring Institution ID'] = st.text_input('Acquiring Institution ID')
user_inputs['Merchant Identifier'] = st.text_input('Merchant Identifier')
user_inputs['Merchant Category Code (MCC)'] = st.text_input('Merchant Category Code (MCC)')


label_encoders = joblib.load('label_encoders.pkl')


encoded_inputs = encode_inputs(user_inputs, label_encoders)


if st.button('Predict Fraud'):
    prediction, probability = predict_fraud(encoded_inputs)
    if prediction == 1:
        st.error(f'Likely Fraud Detected with {probability*100:.2f}% chance of it being a fraud')
    else:
        st.success(f'Unlikely Fraud with {probability*100:.2f}% chance of it being a fraud')
