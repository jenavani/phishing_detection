import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

"""# LOADING THE DATASET AND GETTING THE FEATURES AND CLASS AND SPLITTING THEM INTO TRAINING AND TEST DATAS"""

# Load dataset
df = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv')

# Display the first few rows of the dataset
print(df.head())

# Features and target variable
features = [
    'URLLength', 'DomainLength', 'IsDomainIP', 'URLSimilarityIndex', 'CharContinuationRate',
    'TLDLegitimateProb', 'URLCharProb', 'TLDLength', 'NoOfSubDomain', 'HasObfuscation',
    'NoOfObfuscatedChar', 'ObfuscationRatio', 'NoOfLettersInURL', 'LetterRatioInURL',
    'NoOfDegitsInURL', 'DegitRatioInURL', 'NoOfEqualsInURL', 'NoOfQMarkInURL',
    'NoOfAmpersandInURL', 'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL',
    'IsHTTPS', 'LineOfCode', 'LargestLineLength', 'HasTitle', 'DomainTitleMatchScore',
    'URLTitleMatchScore', 'HasFavicon', 'Robots', 'IsResponsive', 'NoOfURLRedirect',
    'NoOfSelfRedirect', 'HasDescription', 'NoOfPopup', 'NoOfiFrame', 'HasExternalFormSubmit',
    'HasSocialNet', 'HasSubmitButton', 'HasHiddenFields', 'HasPasswordField', 'Bank',
    'Pay', 'Crypto', 'HasCopyrightInfo', 'NoOfImage', 'NoOfCSS', 'NoOfJS',
    'NoOfSelfRef', 'NoOfEmptyRef', 'NoOfExternalRef'
]
target = 'label'

# Separate features and target variable
X = df[features]
y = df[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""# FITTING THE MODEL INTO THE RANDOM CLASSIFIER"""

# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

"""# USING THE MODEL MAKING SOME PREDICTION TO GET THE ACCURACY SCORE"""

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display confusion matrix and classification report
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

print('Classification Report:')
print(classification_report(y_test, y_pred))

"""## CROSS VALIDATING THE ACCURACY SCORE"""

from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean CV Score: {cv_scores.mean()}')

"""# FINE TUNING THE MODEL SO THAT IT WILL GET MORE ACCURACY"""

from sklearn.model_selection import GridSearchCV

# Define hyperparameters
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best parameters
print('Best Parameters:', grid_search.best_params_)

"""# GETTING THE IMPORTANCE OF THE FEATURES"""

importances = model.feature_importances_
feature_importances = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
print('Feature Importances:')
for feature, importance in feature_importances:
    print(f'{feature}: {importance}')

import matplotlib.pyplot as plt
import seaborn as sns

# Calculate feature importances
importances = model.feature_importances_

# Create a DataFrame for feature importances
feature_importances_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances_df)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

"""# FITTING THE MODEL INTO THE RANDOM CLASSIFIER TO IMPROVE THE ACCURACY"""

# Initialize the RandomForestClassifier with the best parameters
optimized_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)

# Train the optimized model
optimized_model.fit(X_train, y_train)

"""# AGAIN PREDICTIONG THE MODEL ACCURACY TO INCREASE ITS EFFICIENCY"""

# Make predictions with the optimized model
y_pred_optimized = optimized_model.predict(X_test)

# Calculate accuracy
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
print(f'Optimized Accuracy: {accuracy_optimized:.2f}')

# Display confusion matrix and classification report
print('Optimized Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_optimized))

print('Optimized Classification Report:')
print(classification_report(y_test, y_pred_optimized))

"""# SAVING AND LOADING THE MODEL TO MAKE PREDICTIONS"""

import joblib

# Save the model
joblib.dump(optimized_model, 'optimized_model.joblib')

# To load the model later
loaded_model = joblib.load('optimized_model.joblib')

from sklearn.preprocessing import StandardScaler
import joblib

# Assuming you have a scaler fitted previously
scaler = StandardScaler()
scaler.fit(X_train)  # Fit on the training data

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')

import pickle

# Save the model
with open('optimized_model.pkl', 'wb') as file:
    pickle.dump(optimized_model, file)

# To load the model later
with open('optimized_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

"""# MAKING THE FEATURE EXTRACTION TO FIND THE PREDICTION ERRORS"""

import joblib

# Load the trained model and scaler
loaded_model = joblib.load('optimized_model.joblib')
loaded_scaler = joblib.load('scaler.joblib')  # Ensure the file exists and is named correctly

# Example new data (Replace this with your own data)
import pandas as pd

new_data = pd.DataFrame({
    'URLLength': [50],
    'DomainLength': [10],
    'IsDomainIP': [0],
    'URLSimilarityIndex': [0.5],
    'CharContinuationRate': [0.02],
    'TLDLegitimateProb': [0.9],
    'URLCharProb': [0.6],
    'TLDLength': [3],
    'NoOfSubDomain': [2],
    'HasObfuscation': [0],
    'NoOfObfuscatedChar': [0],
    'ObfuscationRatio': [0.0],
    'NoOfLettersInURL': [15],
    'LetterRatioInURL': [0.8],
    'NoOfDegitsInURL': [5],
    'DegitRatioInURL': [0.1],
    'NoOfEqualsInURL': [0],
    'NoOfQMarkInURL': [0],
    'NoOfAmpersandInURL': [1],
    'NoOfOtherSpecialCharsInURL': [2],
    'SpacialCharRatioInURL': [0.05],
    'IsHTTPS': [1],
    'LineOfCode': [100],
    'LargestLineLength': [80],
    'HasTitle': [1],
    'DomainTitleMatchScore': [0.9],
    'URLTitleMatchScore': [0.8],
    'HasFavicon': [1],
    'Robots': [0],
    'IsResponsive': [1],
    'NoOfURLRedirect': [1],
    'NoOfSelfRedirect': [0],
    'HasDescription': [1],
    'NoOfPopup': [0],
    'NoOfiFrame': [0],
    'HasExternalFormSubmit': [1],
    'HasSocialNet': [1],
    'HasSubmitButton': [1],
    'HasHiddenFields': [0],
    'HasPasswordField': [0],
    'Bank': [0],
    'Pay': [0],
    'Crypto': [0],
    'HasCopyrightInfo': [1],
    'NoOfImage': [5],
    'NoOfCSS': [3],
    'NoOfJS': [2],
    'NoOfSelfRef': [1],
    'NoOfEmptyRef': [0],
    'NoOfExternalRef': [10]
})

# Preprocess the new data using the loaded scaler
new_data_scaled = loaded_scaler.transform(new_data)

# Predict using the loaded model
prediction = loaded_model.predict(new_data_scaled)

# Display the prediction
print(f"Prediction for new data: {'Phishing' if prediction[0] == 1 else 'Legitimate'}")

from urllib.parse import urlparse
import re
import requests
from bs4 import BeautifulSoup

def extract_features(url):
    features = {}
    parsed_url = urlparse(url)

    # Extract features from URL
    features['URLLength'] = len(url)
    features['DomainLength'] = len(parsed_url.netloc)
    features['IsDomainIP'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', parsed_url.netloc) else 0
    features['TLDLength'] = len(parsed_url.path.split('.')[-1])
    features['NoOfSubDomain'] = len(parsed_url.netloc.split('.')) - 2

    # Placeholders for complex features
    features['URLSimilarityIndex'] = 100.0
    features['CharContinuationRate'] = 1.0
    features['TLDLegitimateProb'] = 0.5
    features['URLCharProb'] = 0.5
    features['HasObfuscation'] = 0
    features['NoOfObfuscatedChar'] = 0
    features['ObfuscationRatio'] = 0.0
    features['NoOfLettersInURL'] = len(re.findall(r'[a-zA-Z]', url))
    features['LetterRatioInURL'] = len(re.findall(r'[a-zA-Z]', url)) / len(url) if len(url) > 0 else 0
    features['NoOfDegitsInURL'] = len(re.findall(r'\d', url))
    features['DegitRatioInURL'] = len(re.findall(r'\d', url)) / len(url) if len(url) > 0 else 0
    features['NoOfEqualsInURL'] = url.count('=')
    features['NoOfQMarkInURL'] = url.count('?')
    features['NoOfAmpersandInURL'] = url.count('&')
    features['NoOfOtherSpecialCharsInURL'] = len(re.findall(r'[^\w\s]', url)) - (features['NoOfEqualsInURL'] + features['NoOfQMarkInURL'] + features['NoOfAmpersandInURL'])
    features['SpacialCharRatioInURL'] = features['NoOfOtherSpecialCharsInURL'] / len(url) if len(url) > 0 else 0
    features['IsHTTPS'] = 1 if parsed_url.scheme == 'https' else 0

    # Fetch and parse HTML content
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        features['LineOfCode'] = len(soup.prettify().split('\n'))
        features['LargestLineLength'] = max(len(line) for line in soup.prettify().split('\n'))
        features['HasTitle'] = 1 if soup.title else 0
        features['DomainTitleMatchScore'] = 0.0
        features['URLTitleMatchScore'] = 0.0
        features['HasFavicon'] = 1 if soup.find('link', rel='icon') else 0
        features['Robots'] = 1 if soup.find('meta', attrs={'name': 'robots'}) else 0
        features['IsResponsive'] = 1 if soup.find('meta', attrs={'name': 'viewport'}) else 0
        features['NoOfURLRedirect'] = 0  # Placeholder
        features['NoOfSelfRedirect'] = len(re.findall(r'#', response.text))
        features['HasDescription'] = 1 if soup.find('meta', attrs={'name': 'description'}) else 0
        features['NoOfPopup'] = len(re.findall(r'alert\(', response.text))
        features['NoOfiFrame'] = len(soup.find_all('iframe'))
        features['HasExternalFormSubmit'] = 1 if any(form.get('action', '').startswith('http') for form in soup.find_all('form')) else 0
        features['HasSocialNet'] = len(soup.find_all(href=re.compile(r'social'))) > 0
        features['HasSubmitButton'] = len(soup.find_all('input', type='submit')) > 0
        features['HasHiddenFields'] = len(soup.find_all('input', type='hidden')) > 0
        features['HasPasswordField'] = len(soup.find_all('input', type='password')) > 0
        features['Bank'] = 0
        features['Pay'] = 0
        features['Crypto'] = 0
        features['HasCopyrightInfo'] = 1 if soup.find(text=re.compile(r'©')) else 0
        features['NoOfImage'] = len(soup.find_all('img'))
        features['NoOfCSS'] = len(soup.find_all('link', rel='stylesheet'))
        features['NoOfJS'] = len(soup.find_all('script'))
        features['NoOfSelfRef'] = len(re.findall(r'#', response.text))
        features['NoOfEmptyRef'] = len(re.findall(r'href=[\'"]?$', response.text))
        features['NoOfExternalRef'] = len(soup.find_all(href=re.compile(r'^http')))

    except Exception as e:
        print(f"Error processing URL: {e}")
        # Set features to default or placeholder values in case of errors
        features.update({
            'LineOfCode': 0,
            'LargestLineLength': 0,
            'HasTitle': 0,
            'DomainTitleMatchScore': 0.0,
            'URLTitleMatchScore': 0.0,
            'HasFavicon': 0,
            'Robots': 0,
            'IsResponsive': 0,
            'NoOfURLRedirect': 0,
            'NoOfSelfRedirect': 0,
            'HasDescription': 0,
            'NoOfPopup': 0,
            'NoOfiFrame': 0,
            'HasExternalFormSubmit': 0,
            'HasSocialNet': 0,
            'HasSubmitButton': 0,
            'HasHiddenFields': 0,
            'HasPasswordField': 0,
            'Bank': 0,
            'Pay': 0,
            'Crypto': 0,
            'HasCopyrightInfo': 0,
            'NoOfImage': 0,
            'NoOfCSS': 0,
            'NoOfJS': 0,
            'NoOfSelfRef': 0,
            'NoOfEmptyRef': 0,
            'NoOfExternalRef': 0,
        })

    return features

import joblib

# Load the scaler
scaler = joblib.load('scaler.joblib')

def preprocess_features(features):
    df = pd.DataFrame([features])
    return scaler.transform(df)

import joblib

# Load the model
model = joblib.load('optimized_model.joblib')

def predict_url(url):
    features = extract_features(url)
    features_scaled = preprocess_features(features)
    prediction = model.predict(features_scaled)
    return 'Phishing' if prediction[0] == 1 else 'Legitimate'

#pip install tldextract

import requests
from bs4 import BeautifulSoup
import tldextract
import joblib
import numpy as np
import pandas as pd

# Load the trained model and scaler
loaded_model = joblib.load('optimized_model.joblib')
loaded_scaler = joblib.load('scaler.joblib')

def extract_features_from_url(url):
    extracted = tldextract.extract(url)
    domain = extracted.domain + '.' + extracted.suffix
    url_length = len(url)
    domain_length = len(domain)
    is_domain_ip = 1 if extracted.domain.replace('.', '').isdigit() else 0
    # Simulate some example features; adjust these as needed based on your feature set
    features = {
        'URLLength': url_length,
        'DomainLength': domain_length,
        'IsDomainIP': is_domain_ip,
        'URLSimilarityIndex': 0.5,
        'CharContinuationRate': 0.02,
        'TLDLegitimateProb': 0.9,
        'URLCharProb': 0.6,
        'TLDLength': len(extracted.suffix),
        'NoOfSubDomain': len(extracted.subdomain.split('.')) if extracted.subdomain else 0,
        'HasObfuscation': 0,
        'NoOfObfuscatedChar': 0,
        'ObfuscationRatio': 0.0,
        'NoOfLettersInURL': sum(c.isalpha() for c in url),
        'LetterRatioInURL': sum(c.isalpha() for c in url) / len(url),
        'NoOfDegitsInURL': sum(c.isdigit() for c in url),
        'DegitRatioInURL': sum(c.isdigit() for c in url) / len(url),
        'NoOfEqualsInURL': url.count('='),
        'NoOfQMarkInURL': url.count('?'),
        'NoOfAmpersandInURL': url.count('&'),
        'NoOfOtherSpecialCharsInURL': sum(not c.isalnum() for c in url),
        'SpacialCharRatioInURL': sum(not c.isalnum() for c in url) / len(url),
        'IsHTTPS': 1 if url.startswith('https') else 0,
        'LineOfCode': 100,  # Placeholder value
        'LargestLineLength': 80,  # Placeholder value
        'HasTitle': 1,  # Placeholder value
        'DomainTitleMatchScore': 0.9,  # Placeholder value
        'URLTitleMatchScore': 0.8,  # Placeholder value
        'HasFavicon': 1,  # Placeholder value
        'Robots': 0,  # Placeholder value
        'IsResponsive': 1,  # Placeholder value
        'NoOfURLRedirect': 1,  # Placeholder value
        'NoOfSelfRedirect': 0,  # Placeholder value
        'HasDescription': 1,  # Placeholder value
        'NoOfPopup': 0,  # Placeholder value
        'NoOfiFrame': 0,  # Placeholder value
        'HasExternalFormSubmit': 1,  # Placeholder value
        'HasSocialNet': 1,  # Placeholder value
        'HasSubmitButton': 1,  # Placeholder value
        'HasHiddenFields': 0,  # Placeholder value
        'HasPasswordField': 0,  # Placeholder value
        'Bank': 0,  # Placeholder value
        'Pay': 0,  # Placeholder value
        'Crypto': 0,  # Placeholder value
        'HasCopyrightInfo': 1,  # Placeholder value
        'NoOfImage': 5,  # Placeholder value
        'NoOfCSS': 3,  # Placeholder value
        'NoOfJS': 2,  # Placeholder value
        'NoOfSelfRef': 1,  # Placeholder value
        'NoOfEmptyRef': 0,  # Placeholder value
        'NoOfExternalRef': 10  # Placeholder value
    }
    return features

def predict_url(url):
    # Extract features
    features = extract_features_from_url(url)

    # Ensure features are in the same order and all are included
    required_features = [
        'URLLength', 'DomainLength', 'IsDomainIP', 'URLSimilarityIndex', 'CharContinuationRate',
        'TLDLegitimateProb', 'URLCharProb', 'TLDLength', 'NoOfSubDomain', 'HasObfuscation',
        'NoOfObfuscatedChar', 'ObfuscationRatio', 'NoOfLettersInURL', 'LetterRatioInURL',
        'NoOfDegitsInURL', 'DegitRatioInURL', 'NoOfEqualsInURL', 'NoOfQMarkInURL',
        'NoOfAmpersandInURL', 'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL',
        'IsHTTPS', 'LineOfCode', 'LargestLineLength', 'HasTitle', 'DomainTitleMatchScore',
        'URLTitleMatchScore', 'HasFavicon', 'Robots', 'IsResponsive', 'NoOfURLRedirect',
        'NoOfSelfRedirect', 'HasDescription', 'NoOfPopup', 'NoOfiFrame', 'HasExternalFormSubmit',
        'HasSocialNet', 'HasSubmitButton', 'HasHiddenFields', 'HasPasswordField', 'Bank',
        'Pay', 'Crypto', 'HasCopyrightInfo', 'NoOfImage', 'NoOfCSS', 'NoOfJS',
        'NoOfSelfRef', 'NoOfEmptyRef', 'NoOfExternalRef'
    ]

    # Arrange features correctly
    data = np.array([features[feature] for feature in required_features]).reshape(1, -1)

    # Preprocess the new data using the loaded scaler
    data_scaled = loaded_scaler.transform(data)

    # Predict using the loaded model
    prediction = loaded_model.predict(data_scaled)

    return 'Phishing' if prediction[0] == 1 else 'Legitimate'

# Example usage
url_to_check = 'https://www.example.com'
result = predict_url(url_to_check)
print(f'The URL "{url_to_check}" is predicted to be: {result}')

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
loaded_model = joblib.load('optimized_model.joblib')
loaded_scaler = joblib.load('scaler.joblib')

# Function to extract features from a URL
def extract_features_from_url(url):
    # This is a placeholder. You'll need to implement actual feature extraction based on your training data.
    # Here is an example structure of the features with dummy data:
    data = {
        'URLLength': [len(url)],
        'DomainLength': [len(url.split("//")[-1].split("/")[0])],
        'IsDomainIP': [1 if url.split("//")[-1].split("/")[0].replace('.', '').isdigit() else 0],
        'URLSimilarityIndex': [0.5],  # Placeholder/ value
        'CharContinuationRate': [0.02],  # Placeholder value
        'TLDLegitimateProb': [0.9],  # Placeholder value
        'URLCharProb': [0.6],  # Placeholder value
        'TLDLength': [len(url.split('.')[-1])],
        'NoOfSubDomain': [url.count('.') - 1],
        'HasObfuscation': [0],  # Placeholder value
        'NoOfObfuscatedChar': [0],  # Placeholder value
        'ObfuscationRatio': [0.0],  # Placeholder value
        'NoOfLettersInURL': [sum(c.isalpha() for c in url)],
        'LetterRatioInURL': [sum(c.isalpha() for c in url) / len(url)],
        'NoOfDegitsInURL': [sum(c.isdigit() for c in url)],
        'DegitRatioInURL': [sum(c.isdigit() for c in url) / len(url)],
        'NoOfEqualsInURL': [url.count('=')],
        'NoOfQMarkInURL': [url.count('?')],
        'NoOfAmpersandInURL': [url.count('&')],
        'NoOfOtherSpecialCharsInURL': [sum(not c.isalnum() for c in url) - (url.count('&') + url.count('?') + url.count('='))],
        'SpacialCharRatioInURL': [sum(not c.isalnum() for c in url) / len(url)],
        'IsHTTPS': [1 if url.startswith('https') else 0],
        'LineOfCode': [100],  # Placeholder value
        'LargestLineLength': [80],  # Placeholder value
        'HasTitle': [1],  # Placeholder value
        'DomainTitleMatchScore': [0.9],  # Placeholder value
        'URLTitleMatchScore': [0.8],  # Placeholder value
        'HasFavicon': [1],  # Placeholder value
        'Robots': [0],  # Placeholder value
        'IsResponsive': [1],  # Placeholder value
        'NoOfURLRedirect': [1],  # Placeholder value
        'NoOfSelfRedirect': [0],  # Placeholder value
        'HasDescription': [1],  # Placeholder value
        'NoOfPopup': [0],  # Placeholder value
        'NoOfiFrame': [0],  # Placeholder value
        'HasExternalFormSubmit': [1],  # Placeholder value
        'HasSocialNet': [1],  # Placeholder value
        'HasSubmitButton': [1],  # Placeholder value
        'HasHiddenFields': [0],  # Placeholder value
        'HasPasswordField': [0],  # Placeholder value
        'Bank': [0],  # Placeholder value
        'Pay': [0],  # Placeholder value
        'Crypto': [0],  # Placeholder value
        'HasCopyrightInfo': [1],  # Placeholder value
        'NoOfImage': [5],  # Placeholder value
        'NoOfCSS': [3],  # Placeholder value
        'NoOfJS': [2],  # Placeholder value
        'NoOfSelfRef': [1],  # Placeholder value
        'NoOfEmptyRef': [0],  # Placeholder value
        'NoOfExternalRef': [10]  # Placeholder value
    }

    return pd.DataFrame(data)

# Function to preprocess and predict URL
def predict_url(url):
    # Extract features
    features = extract_features_from_url(url)

    # Ensure features are in the same order as the training data
    feature_order = [
        'URLLength', 'DomainLength', 'IsDomainIP', 'URLSimilarityIndex', 'CharContinuationRate',
        'TLDLegitimateProb', 'URLCharProb', 'TLDLength', 'NoOfSubDomain', 'HasObfuscation',
        'NoOfObfuscatedChar', 'ObfuscationRatio', 'NoOfLettersInURL', 'LetterRatioInURL',
        'NoOfDegitsInURL', 'DegitRatioInURL', 'NoOfEqualsInURL', 'NoOfQMarkInURL',
        'NoOfAmpersandInURL', 'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL',
        'IsHTTPS', 'LineOfCode', 'LargestLineLength', 'HasTitle', 'DomainTitleMatchScore',
        'URLTitleMatchScore', 'HasFavicon', 'Robots', 'IsResponsive', 'NoOfURLRedirect',
        'NoOfSelfRedirect', 'HasDescription', 'NoOfPopup', 'NoOfiFrame', 'HasExternalFormSubmit',
        'HasSocialNet', 'HasSubmitButton', 'HasHiddenFields', 'HasPasswordField', 'Bank',
        'Pay', 'Crypto', 'HasCopyrightInfo', 'NoOfImage', 'NoOfCSS', 'NoOfJS',
        'NoOfSelfRef', 'NoOfEmptyRef', 'NoOfExternalRef'
    ]
    features = features[feature_order]

    # Preprocess the data
    features_scaled = loaded_scaler.transform(features.to_numpy())  # Convert to numpy array to avoid warning

    # Predict using the loaded model
    prediction = loaded_model.predict(features_scaled)

    # Display the prediction
    print(f"Prediction for URL '{url}': {'Phishing' if prediction[0] == 1 else 'Legitimate'}")

# Example usage
url = "https://apple.ruipaicn.com"
predict_url(url)