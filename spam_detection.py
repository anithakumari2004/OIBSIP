
# STEP 1: IMPORT NECESSARY LIBRARIES

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# STEP 2: LOAD AND PREPARE THE DATA

file_path = r"C:\Users\anita\OneDrive\Desktop\spam_prediction.csv"
try:
    
    df = pd.read_csv(file_path, encoding='utf-8-sig')
except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
    exit()
df = df[['v1', 'v2']]
df.columns = ['Category', 'Message']
print("## First 5 rows of the cleaned dataset:")
print(df.head())
df['Category'] = df['Category'].map({'spam': 1, 'ham': 0})
print("\n## Dataset Information:")
df.info()

# STEP 3: DEFINE FEATURES (X) AND TARGET (Y)
X = df['Message']
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# STEP 4: FEATURE EXTRACTION (TEXT TO VECTORS)
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# STEP 5: TRAIN THE SPAM DETECTION MODEL
model = MultinomialNB()
print("\n## Training the model...")
model.fit(X_train_tfidf, y_train)
print("## Model training complete!")

# STEP 6: EVALUATE THE MODEL
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n## Model Accuracy: {accuracy:.4f}")

print("\n## Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

print("\n## Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# STEP 7: TEST THE MODEL WITH NEW EMAILS

def predict_spam(email_text):
    email_tfidf = vectorizer.transform([email_text])
    prediction = model.predict(email_tfidf)
    return "Spam" if prediction[0] == 1 else "Not Spam (Ham)"

email1 = "Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/spam-link to claim now."
email2 = "Hey, are we still on for the meeting tomorrow at 2 PM? Let me know."

print("\n## Testing with new emails:")
print(f"Email: '{email1}' \nPrediction: {predict_spam(email1)}\n")
print(f"Email: '{email2}' \nPrediction: {predict_spam(email2)}")
