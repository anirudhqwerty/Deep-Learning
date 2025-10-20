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

Install all the required Python packages.

```bash
pip install scikit-learn nltk streamlit PyPDF2 docx2txt
```

### 4\. Run the Application

Start the Streamlit web server.

```bash
streamlit run app.py
```

Once the server is running, navigate to the local URL provided in your terminal (usually `http://localhost:8501`). You can then upload a resume or paste text to see the predictions.

**Note on the Dataset:** The pre-trained models (`resume_model.pkl` and `tfidf_vectorizer.pkl`) allow the application to run without the original dataset. The `Preprocessed_data.txt` file, which is over 50MB, is not included in this repository. If you wish to retrain the model or explore the data, you can download it from **[\[Dataset\]](https://github.com/noran-mohamed/Resume-Classification-Dataset)** and place it in the root project folder.

-----

## 🧐 Limitations & Future Improvements

While effective for its purpose, this project has some inherent limitations due to the classic NLP approach used.

-   **Keyword Dependency:** The model's core logic is based on TF-IDF, which prioritizes keyword frequency over semantic context. This can sometimes lead to inaccurate predictions if a resume contains keywords that overlap with an unrelated job category.
-   **No Semantic Understanding:** The model does not understand the meaning or relationships between words. It cannot infer skills or experience from context, which is a key limitation compared to more advanced deep learning models.
-   **Potential Data Bias:** Job categories that are underrepresented in the training dataset may not be predicted as accurately as more common categories.

### Future Work

To address these limitations, future versions of this project could incorporate:

-   **Advanced Embeddings:** Replace TF-IDF with contextual word embeddings from models like **BERT**, **Sentence-BERT**, or **spaCy** to achieve a deeper semantic understanding of the resume text.
-   **Model Enhancement:** Experiment with more complex architectures, such as neural networks, to better capture the nuances of skills and experiences described in a resume.

## 👤 Author

**Anirudh Sharma**

  - Computer Engineering, Thapar Institute of Technology, Patiala
  - **GitHub:** [@anirudhqwerty](https://github.com/anirudhqwerty)
  - **Email:** `therealanirudhsharma@gmail.com`