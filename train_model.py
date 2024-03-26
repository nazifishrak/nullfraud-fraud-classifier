import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv('BOLT Data Set.csv')
df['Risk Assessment'].fillna(df['Risk Assessment'].mean(), inplace=True)
df['Merchant Category Code (MCC)'].fillna(df['Merchant Category Code (MCC)'].mode()[0], inplace=True)

label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != 'Fraud Indicator (Yes/No)':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

df['Fraud Indicator (Yes/No)'] = df['Fraud Indicator (Yes/No)'].apply(lambda x: 1 if x == 'Yes' else 0)

oversample_instances = df[(df['Card Present Status'] == 0)]
oversample_factor = 2  
oversampled_data = pd.concat([oversample_instances] * oversample_factor, ignore_index=True)
df_oversampled = pd.concat([df, oversampled_data], ignore_index=True)
df = df_oversampled

X = df.drop(['Fraud Indicator (Yes/No)'], axis=1)
y = df['Fraud Indicator (Yes/No)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

feature_order = X_train.columns.tolist()

joblib.dump(feature_order, 'feature_order.pkl')

rf_classifier_balanced = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_classifier_balanced.fit(X_train, y_train)

joblib.dump(rf_classifier_balanced, 'rf_classifier_balanced.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

y_pred_balanced = rf_classifier_balanced.predict(X_test)


def decode_inputs(encoded_inputs, label_encoders, original_columns):
    decoded_inputs = {}
    for col in original_columns:
        if col in label_encoders:
            le = label_encoders[col]
            encoded_value = encoded_inputs[col]
            if isinstance(encoded_value, float) and encoded_value.is_integer():
                encoded_value = int(encoded_value)
            decoded_inputs[col] = le.inverse_transform([encoded_value])[0]
    return decoded_inputs


fraud_indices = np.where(y_pred_balanced == 1)[0]
fraud_instances = X_test.iloc[fraud_indices]

original_columns = X_test.columns


decoded_fraud_instances = []
for _, row in fraud_instances.iterrows():
    row_dict = row.to_dict()
    decoded_row = decode_inputs(row_dict, label_encoders, original_columns)
    decoded_fraud_instances.append(decoded_row)

decoded_fraud_df = pd.DataFrame(decoded_fraud_instances, columns=original_columns)
print("Decoded features of instances classified as fraud:")
print(decoded_fraud_df)


accuracy_balanced = accuracy_score(y_test, y_pred_balanced)


final_class_report = classification_report(y_test, y_pred_balanced)
print('Accuracy:', accuracy_balanced)
print(final_class_report)
