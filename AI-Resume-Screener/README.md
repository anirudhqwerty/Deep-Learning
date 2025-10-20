Of course. I've restructured and refined your project's `README.md` to be more professional, clear, and comprehensive.

Here is the revised version:

-----

# AI Resume Screener

An NLP-based web application built with Python and Streamlit that analyzes resumes to identify the top 3 best-matching job categories. This tool provides a similarity score for each predicted role, helping both recruiters and job seekers quickly assess resume suitability.

-----

## 🚀 Features

  - **File Upload & Text Input:** Accepts resumes in both `.pdf` and `.docx` formats, or you can paste raw text directly.
  - **Job Category Prediction:** Utilizes a trained machine learning model to predict the top 3 most relevant job categories for a given resume.
  - **Match Score Visualization:** Displays a percentage-based match score for each predicted category using an interactive bar chart.
  - **Broad Coverage:** Supports over 50 different job categories.
  - **Easy to Deploy:** A lightweight Streamlit application that is simple to run locally or deploy.

-----

## ⚙️ How It Works

The screening process is powered by classic Natural Language Processing (NLP) techniques:

1.  **Text Extraction:** Resumes are parsed to extract raw text using `PyPDF2` for PDFs and `docx2txt` for Word documents.
2.  **Text Preprocessing:** The extracted text is cleaned by removing stopwords and other noise using `NLTK`.
3.  **Feature Engineering:** The cleaned text is converted into a numerical vector representation using a pre-trained `TF-IDF Vectorizer`.
4.  **Prediction:** A `Logistic Regression` model, trained on a labeled dataset of resumes, predicts the most likely job category.
5.  **Scoring:** `Cosine Similarity` is used to calculate the match score between the input resume's TF-IDF vector and the vectors representing each of the top predicted categories.

-----

## 🛠️ Tech Stack

  - **Backend:** Python 3
  - **Machine Learning:** Scikit-Learn (TF-IDF, Logistic Regression, Cosine Similarity)
  - **NLP:** NLTK
  - **Web Framework:** Streamlit
  - **Data Handling:** Pandas, Numpy
  - **File Parsing:** PyPDF2, docx2txt

-----

## 📂 Project Structure

```
Resume-Screener/
├── app.py                 # Main Streamlit application file
├── resume_model.pkl       # Pre-trained Logistic Regression model
├── tfidf_vectorizer.pkl   # Pre-trained TF-IDF vectorizer
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── Preprocessed_data.txt  # (Optional) Dataset for retraining
```

-----

## ⚡ Getting Started

Follow these steps to run the project locally.

### 1\. Prerequisites

  - Python 3.7+
  - `pip` package manager

### 2\. Clone the Repository

```bash
git clone https://github.com/anirudhqwerty/Resume-Screener.git
cd Resume-Screener
```

### 3\. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4\. Run the Application

Start the Streamlit web server.

```bash
streamlit run app.py
```

Once the server is running, navigate to the local URL provided in your terminal (usually `http://localhost:8501`). You can then upload a resume or paste text to see the predictions.

**Note on the Dataset:** The pre-trained models (`resume_model.pkl` and `tfidf_vectorizer.pkl`) allow the application to run without the original dataset. The `Preprocessed_data.txt` file, which is over 50MB, is not included in this repository. If you wish to retrain the model or explore the data, you can download it from **[\[INSERT DOWNLOAD LINK HERE\]](https://github.com/noran-mohamed/Resume-Classification-Dataset)** and place it in the root project folder.

-----

## 👤 Author

**Anirudh Sharma**

  - Computer Engineering, Thapar Institute of Technology, Patiala
  - **GitHub:** [@anirudhqwerty](https://www.google.com/search?q=https://github.com/anirudhqwerty)
  - **Email:** `therealanirudhsharma@gmail.com`