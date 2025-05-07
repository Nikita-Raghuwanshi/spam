import sys
import joblib
import numpy as np

# Load the pre-trained models and vectorizer
nb_model = joblib.load('./nb_model.pkl')  # Adjust the path if necessary
vectorizer = joblib.load('./vectorizer.pkl')
label_encoder = joblib.load('./label_encoder.pkl')

# Get the email text from command-line argument
email_text = sys.argv[1]

# Transform the email text using the vectorizer
X_input = vectorizer.transform([email_text]).toarray()

# Make a prediction
nb_pred = nb_model.predict(X_input)

# Debugging: Check prediction
print("Raw numeric prediction:", nb_pred[0])

# Decode the prediction to human-readable form
prediction_label = label_encoder.inverse_transform([nb_pred[0]])[0]

# Output the prediction
print(f"Prediction: {prediction_label}")  # Should print "spam" or "not spam"
