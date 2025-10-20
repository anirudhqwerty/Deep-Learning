# resume screener web app - top 3 categories + match score

import streamlit as st
import pandas as pd
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ------------------------------
# text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# ------------------------------
# load model and vectorizer
model = joblib.load("resume_model.pkl")       # your trained logistic regression
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # your trained tfidf

# ------------------------------
# load job descriptions per category
# (combine all resumes per category)
df = pd.read_csv("Preprocessed_data.txt")
df.columns = ['Category', 'Text']
job_descriptions = df.groupby("Category")["Text"].apply(lambda x: " ".join(x)).to_dict()

# ------------------------------
# function to get top 3 matches
def get_top3_matches(resume_text):
    resume_clean = clean_text(resume_text)
    scores = {}
    for cat, jd in job_descriptions.items():
        vectors = vectorizer.transform([resume_clean, jd])
        score = cosine_similarity(vectors[0], vectors[1])[0][0]
        scores[cat] = round(score*100, 2)
    # sort by score descending
    top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    return top3

# ------------------------------
# streamlit UI
st.title("🧠 AI Resume Screener")
st.write("Upload a resume file (.pdf/.docx) or paste text to see top 3 matching job categories")

# upload resume
resume_file = st.file_uploader("Upload Resume", type=["pdf","docx"])
resume_text_area = st.text_area("Or paste resume text here")

if st.button("Check Top 3 Categories"):
    # get resume text
    resume_text = ""
    if resume_file:
        import docx2txt, PyPDF2
        if resume_file.name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(resume_file)
            for page in reader.pages:
                resume_text += page.extract_text()
        elif resume_file.name.endswith(".docx"):
            resume_text = docx2txt.process(resume_file)
    elif resume_text_area:
        resume_text = resume_text_area
    else:
        st.warning("Please upload a file or paste text")
    
    if resume_text:
        top3 = get_top3_matches(resume_text)
        st.subheader("Top 3 Matching Categories")
        for i, (cat, score) in enumerate(top3, start=1):
            st.write(f"{i}. **{cat}** - Match Score: {score}%")
            st.progress(score/100)
