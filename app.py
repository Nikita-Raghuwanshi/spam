import os
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load models and vectorizer
nb_model = joblib.load('./nb_model.pkl')
vectorizer = joblib.load('./vectorizer.pkl')
label_encoder = joblib.load('./label_encoder.pkl')

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Spam Detection API. Use POST /predict with a JSON payload like {'message': 'Your email here'}."})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    email_text = data.get('message', '')

    if not email_text:
        return jsonify({'error': 'Empty input'}), 400

    X_input = vectorizer.transform([email_text]).toarray()
    nb_pred = nb_model.predict(X_input)
    prediction_label = label_encoder.inverse_transform([nb_pred[0]])[0]

    return jsonify({'prediction': prediction_label})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port)
