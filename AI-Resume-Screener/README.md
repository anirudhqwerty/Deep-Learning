# 🧠 AI Resume Screener

**Check which job roles your resume best matches!**  

This is a **NLP-based resume screening web app** built with Python and Streamlit. It predicts the **top 3 job categories** for a resume and shows a **match score** for each category. No deep learning required — just classic NLP with TF-IDF and cosine similarity.

---

## Features

- Upload a **resume (.pdf/.docx)** or paste text  
- Predict **top 3 job categories** with match score  
- Interactive **match score bar** for visualization  
- Works with **50+ job categories**  
- Easy to **deploy and share**  

---

## Tech Stack

- Python 3  
- Pandas, Numpy  
- Scikit-Learn (TF-IDF + Logistic Regression)  
- NLTK (stopwords removal)  
- Streamlit (web app)  
- PyPDF2, docx2txt (resume file reading)  

---

## Project Structure

Resume-Screener/
├── Preprocessed_data.txt # Dataset (not included on GitHub due to size)
├── resume_model.pkl # Trained ML model
├── tfidf_vectorizer.pkl # TF-IDF vectorizer
├── app.py # Streamlit app
├── README.md # Project documentation
├── requirements.txt # Required Python packages

yaml
Copy code

---

## How to Run

1. Clone the repo  
```bash
git clone https://github.com/anirudhqwerty/Resume-Screener.git
cd Resume-Screener
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the web app

bash
Copy code
streamlit run app.py
Upload your resume or paste text → see top 3 categories + match scores

Note: Preprocessed_data.txt is not included in the repo due to size (>50MB). Download it separately and place it in the project folder.
https://github.com/noran-mohamed/Resume-Classification-Dataset

Author
Anirudh Sharma
Computer Engineering, Thapar Institute of Technology, Patiala

GitHub: github.com/anirudhqwerty

Email: therealanirudhsharma@gmail.com