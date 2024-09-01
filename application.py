from flask import Flask, render_template, request, jsonify
import joblib
import tldextract
import numpy as np
import pandas as pd
from urllib.parse import urlparse
import re
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load the trained model and scaler
model = joblib.load('optimized_model.joblib')
scaler = joblib.load('scaler.joblib')

def extract_features_from_url(url):
    try:
        parsed_url = urlparse(url)
        extracted = tldextract.extract(url)
        domain = extracted.domain + '.' + extracted.suffix

        # Initialize features with default values
        features = {
            'URLLength': len(url),
            'DomainLength': len(domain),
            'IsDomainIP': 1 if re.match(r'\d+\.\d+\.\d+\.\d+', domain) else 0,
            'URLSimilarityIndex': len(set(url)) / len(url),
            'CharContinuationRate': len(re.findall(r'(.)\1+', url)) / len(url),
            'TLDLegitimateProb': 0.9 if extracted.suffix in ['com', 'org', 'net', 'edu', 'gov'] else 0.1,
            'URLCharProb': len(re.findall(r'[a-zA-Z0-9-.]', url)) / len(url),
            'TLDLength': len(extracted.suffix),
            'NoOfSubDomain': len(extracted.subdomain.split('.')) if extracted.subdomain else 0,
            'HasObfuscation': 1 if re.search(r'%[0-9a-fA-F]{2}', url) else 0,
            'NoOfObfuscatedChar': len(re.findall(r'%[0-9a-fA-F]{2}', url)),
            'ObfuscationRatio': len(re.findall(r'%[0-9a-fA-F]{2}', url)) / len(url) if len(url) > 0 else 0,
            'NoOfLettersInURL': sum(c.isalpha() for c in url),
            'LetterRatioInURL': sum(c.isalpha() for c in url) / len(url),
            'NoOfDegitsInURL': sum(c.isdigit() for c in url),
            'DegitRatioInURL': sum(c.isdigit() for c in url) / len(url),
            'NoOfEqualsInURL': url.count('='),
            'NoOfQMarkInURL': url.count('?'),
            'NoOfAmpersandInURL': url.count('&'),
            'NoOfOtherSpecialCharsInURL': sum(not c.isalnum() for c in url) - (url.count('=') + url.count('?') + url.count('&')),
            'SpacialCharRatioInURL': sum(not c.isalnum() for c in url) / len(url),
            'IsHTTPS': 1 if parsed_url.scheme == 'https' else 0,
            'LineOfCode': 0,
            'LargestLineLength': 0,
            'HasTitle': 0,
            'DomainTitleMatchScore': 0,
            'URLTitleMatchScore': 0,
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
            'NoOfExternalRef': 0
        }

        try:
            # Fetch the webpage content
            response = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.content, 'html.parser')

            # Update features based on webpage content
            features.update({
                'LineOfCode': len(response.text.splitlines()),
                'LargestLineLength': max(len(line) for line in response.text.splitlines()),
                'HasTitle': 1 if soup.title else 0,
                'DomainTitleMatchScore': 1 if soup.title and domain.lower() in soup.title.string.lower() else 0,
                'URLTitleMatchScore': 1 if soup.title and url.lower() in soup.title.string.lower() else 0,
                'HasFavicon': 1 if soup.find('link', rel='icon') else 0,
                'Robots': 1 if soup.find('meta', attrs={'name': 'robots'}) else 0,
                'IsResponsive': 1 if soup.find('meta', attrs={'name': 'viewport'}) else 0,
                'NoOfURLRedirect': len(response.history),
                'NoOfSelfRedirect': len([r for r in response.history if r.url.startswith(url)]),
                'HasDescription': 1 if soup.find('meta', attrs={'name': 'description'}) else 0,
                'NoOfPopup': len(re.findall(r'window\.open|alert\(', response.text)),
                'NoOfiFrame': len(soup.find_all('iframe')),
                'HasExternalFormSubmit': 1 if any(form.get('action', '').startswith('http') for form in soup.find_all('form')) else 0,
                'HasSocialNet': 1 if soup.find('a', href=re.compile(r'facebook|twitter|instagram|linkedin')) else 0,
                'HasSubmitButton': 1 if soup.find('input', type='submit') else 0,
                'HasHiddenFields': 1 if soup.find('input', type='hidden') else 0,
                'HasPasswordField': 1 if soup.find('input', type='password') else 0,
                'Bank': 1 if re.search(r'bank|credit|debit', response.text, re.I) else 0,
                'Pay': 1 if re.search(r'pay|payment', response.text, re.I) else 0,
                'Crypto': 1 if re.search(r'crypto|bitcoin|ethereum', response.text, re.I) else 0,
                'HasCopyrightInfo': 1 if re.search(r'©|copyright', response.text, re.I) else 0,
                'NoOfImage': len(soup.find_all('img')),
                'NoOfCSS': len(soup.find_all('link', rel='stylesheet')),
                'NoOfJS': len(soup.find_all('script')),
                'NoOfSelfRef': len(soup.find_all('a', href=lambda href: href and not href.startswith('http'))),
                'NoOfEmptyRef': len(soup.find_all('a', href='#')),
                'NoOfExternalRef': len(soup.find_all('a', href=lambda href: href and href.startswith('http')))
            })
        except RequestException as e:
            print(f"Error fetching URL content: {e}")
            # We'll continue with the default values for content-based features

        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def preprocess_features(features):
    df = pd.DataFrame([features])
    return scaler.transform(df)

@app.route('/')
def home():
    return render_template('frontpage.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data['url']
    
    features = extract_features_from_url(url)
    if features is None:
        return jsonify({'error': 'Failed to extract features from the URL'}), 400
    
    features_scaled = preprocess_features(features)
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0]
    result = 'Phishing' if prediction[0] == 1 else 'Legitimate'
    confidence = probability[1] if result == 'Phishing' else probability[0]
    
    return jsonify({
        'result': result,
        'confidence': f"{confidence * 100:.2f}%",
        'features': features
    })

if __name__ == '__main__':
    app.run(debug=True)