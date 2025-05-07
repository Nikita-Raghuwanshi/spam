import pandas as pd
import numpy as np
import nltk
import spacy
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import joblib
from scipy.stats import mode  # Used for proper ensemble voting

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load SpaCy model (optional)
nlp = spacy.load("en_core_web_sm")

# Load the dataset
combined_data = pd.read_csv("C:/Users/nikit/OneDrive/Desktop/spam/combined.csv")

# Handle missing values
combined_data['email_text'] = combined_data['email_text'].fillna("")
combined_data['label'] = combined_data['label'].fillna("unknown")

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

# Compute TF-IDF features
X = vectorizer.fit_transform(combined_data['email_text']).toarray()

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(combined_data['label'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize models
nb_model = MultinomialNB()
lr_model = LogisticRegression(C=100, solver='liblinear', random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

# Fix Upsampling for Imbalanced Classes
df_train = pd.DataFrame(X_train)
df_train['label'] = y_train
df_majority = df_train[df_train['label'] == df_train['label'].mode()[0]]  # Majority class
df_minority = df_train[df_train['label'] != df_train['label'].mode()[0]]  # Minority class

df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_train_resampled = pd.concat([df_majority, df_minority_upsampled])

X_train_resampled = df_train_resampled.drop(columns=['label']).to_numpy()
y_train_resampled = df_train_resampled['label'].to_numpy()

# Train models
print("Training models...")
nb_model.fit(X_train_resampled, y_train_resampled)
lr_model.fit(X_train_resampled, y_train_resampled)
rf_model.fit(X_train_resampled, y_train_resampled)

# Save trained models
joblib.dump(nb_model, 'nb_model.pkl')
joblib.dump(lr_model, 'lr_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("Models saved successfully!")

# Predictions and evaluations
print("\nEvaluating Naive Bayes...")
y_pred_nb = nb_model.predict(X_test)
print(classification_report(y_test, y_pred_nb))

print("\nEvaluating Logistic Regression...")
y_pred_lr = lr_model.predict(X_test)
print(classification_report(y_test, y_pred_lr))

print("\nEvaluating Random Forest...")
y_pred_rf = rf_model.predict(X_test)
print(classification_report(y_test, y_pred_rf))

# Proper Ensemble Voting (Using Majority Vote Instead of Weighted Sum)
def ensemble_voting(predictions):
    predictions_array = np.array(predictions)
    return mode(predictions_array, axis=0).mode.squeeze()  # Ensure proper shape

# Combine predictions using majority voting
predictions = [y_pred_nb, y_pred_lr, y_pred_rf]
y_pred_ensemble = ensemble_voting(predictions)  # Fix applied

# Evaluate ensemble model
print("\nEvaluating Ensemble...")
print(classification_report(y_test, y_pred_ensemble))  # No more error

# Additional Evaluation Metrics (Accuracy and Confusion Matrix)
print("\nAccuracy of Naive Bayes:", accuracy_score(y_test, y_pred_nb))
print("Confusion Matrix for Naive Bayes:\n", confusion_matrix(y_test, y_pred_nb))

print("\nAccuracy of Logistic Regression:", accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix for Logistic Regression:\n", confusion_matrix(y_test, y_pred_lr))

print("\nAccuracy of Random Forest:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix for Random Forest:\n", confusion_matrix(y_test, y_pred_rf))

print("\nAccuracy of Ensemble:", accuracy_score(y_test, y_pred_ensemble))
print("Confusion Matrix for Ensemble:\n", confusion_matrix(y_test, y_pred_ensemble))

# Optional: Use Voting Classifier for ensemble (alternative to majority voting above)
ensemble_model = VotingClassifier(estimators=[
    ('nb', nb_model),
    ('lr', lr_model),
    ('rf', rf_model)
], voting='hard')

# Train the Voting Classifier on the resampled data
ensemble_model.fit(X_train_resampled, y_train_resampled)
y_pred_ensemble_voting = ensemble_model.predict(X_test)

# Evaluate the Voting Classifier
print("\nEvaluating Voting Classifier (Hard Voting)...")
print(classification_report(y_test, y_pred_ensemble_voting))

# Ensure labels are stored correctly as 'spam' and 'not spam'
# Convert numeric labels to text labels
combined_data['label'] = combined_data['label'].astype(str)  # Ensure labels are treated as strings
combined_data['label'] = combined_data['label'].replace({'0': 'not spam', '1': 'spam'})  # Fix labels

# Print unique values before encoding to verify correct labels
print("Unique labels before encoding:", combined_data['label'].unique())  # Should print ['spam' 'not spam']

# Encode labels correctly
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(combined_data['label'])

# Print the mapping of labels to numbers
print("Label Encoder Classes:", label_encoder.classes_)  # Should print ['not spam' 'spam']

# Save LabelEncoder after fixing
joblib.dump(label_encoder, 'label_encoder.pkl')
