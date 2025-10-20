# resume classification using nlp

import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# load data
df = pd.read_csv("Preprocessed_data.txt")
print("data loaded")
print(df.head())

# rename columns
df.columns = ['Category', 'Text']

# remove missing values
df.dropna(inplace=True)

# clean text
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df['Cleaned_Text'] = df['Text'].apply(clean_text)

# convert text to numbers
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Cleaned_Text']).toarray()
y = df['Category']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# test model
y_pred = model.predict(X_test)
print("accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens')
plt.xlabel("predicted")
plt.ylabel("actual")
plt.show()

# save model
joblib.dump(model, "resume_model.pkl")

# save vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("model and vectorizer saved!")

# predict on new resume
sample_resume = """Python developer with experience in data analysis, pandas, numpy, and machine learning.
Worked on Flask API development and REST-based backend systems."""
sample_clean = clean_text(sample_resume)
sample_vec = vectorizer.transform([sample_clean])
prediction = model.predict(sample_vec)
print("predicted category:", prediction[0])
