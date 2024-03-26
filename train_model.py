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

# load and inspect the dataset
df = pd.read_csv('BOLT Data Set.csv')
df['Risk Assessment'].fillna(df['Risk Assessment'].mean(), inplace=True)
df['Merchant Category Code (MCC)'].fillna(df['Merchant Category Code (MCC)'].mode()[0], inplace=True)

# Encode categorical features
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != 'Fraud Indicator (Yes/No)':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Convert target variable to binary
df['Fraud Indicator (Yes/No)'] = df['Fraud Indicator (Yes/No)'].apply(lambda x: 1 if x == 'Yes' else 0)

# Oversampling
oversample_instances = df[(df['Card Present Status'] == 0)]
oversample_factor = 2  
oversampled_data = pd.concat([oversample_instances] * oversample_factor, ignore_index=True)
df_oversampled = pd.concat([df, oversampled_data], ignore_index=True)
df = df_oversampled

# Splitting dataset
X = df.drop(['Fraud Indicator (Yes/No)'], axis=1)
y = df['Fraud Indicator (Yes/No)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Training model
rf_classifier_balanced = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_classifier_balanced.fit(X_train, y_train)

# Saving model and label encoders
joblib.dump(rf_classifier_balanced, 'rf_classifier_balanced.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

y_pred_balanced = rf_classifier_balanced.predict(X_test)


accuracy_balanced = accuracy_score(y_test, y_pred_balanced)


# Print results
final_class_report = classification_report(y_test, y_pred_balanced)
print('Accuracy:', accuracy_balanced)
print(final_class_report)
