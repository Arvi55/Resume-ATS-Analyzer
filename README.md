# 📄 Resume–Job Match Analyzer using NLP & Machine Learning  

*I built an ATS-inspired resume analyzer using NLP techniques like TF-IDF, keyword overlap, and dynamic weighted scoring to realistically evaluate resume–job alignment.*

🚀 **Live Project:** [Click Here to Try the Spam Classifier](https://email-sms-spam-classifier-avinash.streamlit.app/](https://avinash-resume-ats-analyzer.streamlit.app/) 

---

## 🖥️ Web Interface  
Upload a resume (PDF) and paste a job description to instantly get a match score based on ATS-style logic.

![Web Face](https://github.com/Arvi55/Resume-ATS-Analyzer/blob/main/Imgaes/webface%20image.png?raw=true)
---

## 📌 Overview  

The **Resume–Job Match Analyzer** is an NLP-powered web application that evaluates how well a candidate’s resume aligns with a given job description.

Recruiters commonly use **Applicant Tracking Systems (ATS)** to filter resumes based on keyword relevance and contextual similarity.  

This project simulates that behavior by combining **semantic similarity (TF-IDF)** and **keyword overlap analysis**, producing a realistic and explainable match score.

---

## 🧩 Problem Statement  

Recruiters receive hundreds of resumes for a single role, making manual screening inefficient.

Traditional ATS systems:
- Rely heavily on keyword matching  
- Often fail to balance semantic meaning and contextual relevance  

### Project Goals:
- Build an **ATS-inspired resume screening system**
- Compare resumes against job descriptions using NLP
- Generate a **fair, explainable, and realistic match score**

---

## ⚙️ Core Features  

- Upload resume in **PDF format**
- Paste job description text
- Automatic text extraction & cleaning
- **TF-IDF + Cosine Similarity** for semantic matching
- **Keyword overlap analysis** for ATS-style matching
- **Dynamic weighted scoring** based on job description length
- Visual score representation with progress bar

---

## 📂 Input Data  

### Resume Input
- Format: PDF  
- Content: Skills, experience, projects, education  

### Job Description Input
- Format: Plain text  
- Content: Role responsibilities & required skills  

---

## 🔍 Data Preprocessing  

Both resume and job description go through NLP preprocessing before scoring.

### Steps:
- Convert text to lowercase
- Remove special characters and extra spaces
- Tokenize text
- Remove stopwords (e.g., *the, is, and*)
- Normalize text for fair comparison

---

## 🧠 Matching & Scoring Logic  

### 1️⃣ TF-IDF Similarity  
- Converts resume and job description text into numerical vectors
- Measures **semantic similarity** using cosine similarity
- Captures contextual alignment beyond exact keyword matching

---

### 2️⃣ Keyword Overlap Score  
- Extracts meaningful keywords from the job description
- Measures how many JD keywords appear in the resume
- Simulates **ATS keyword-matching behavior**

---

### 3️⃣ Dynamic Weighted Scoring  

Different job descriptions behave differently:

- **Short, skill-based JDs** → Keyword overlap is weighted higher  
- **Long, descriptive JDs** → Semantic similarity is weighted higher  

This ensures:
- Realistic scoring
- No artificial inflation
- Better alignment with real ATS systems

---

## 📊 Output & Visualization  

### Results Displayed:
- **Final Match Score (%)**
- TF-IDF similarity score
- Keyword overlap score
- Color-coded match indicator:
  - 🔴 Low Match
  - 🟠 Good Match
  - 🟢 Excellent Match

---

## 📈 Key Observations  

- Keyword-heavy resumes perform better on short JDs
- Semantic similarity improves scoring for descriptive roles
- Balanced scoring prevents keyword stuffing
- Realistic match scores typically range between **40–70%**, similar to ATS systems

---

## 🛠️ Tech Stack  

- **Python**
- **Streamlit** (Web Interface)
- **Scikit-learn** (TF-IDF & Cosine Similarity)
- **NLTK** (Text preprocessing)
- **PyPDF2** (PDF text extraction)
- **Matplotlib** (Visualization)

---

## 🚀 Future Improvements  

- Highlight **matched vs missing skills**
- Section-wise weighting (Skills > Projects > Education)
- Semantic embeddings using **BERT / Sentence Transformers**
- Resume ranking for multiple candidates
- Deployment on Streamlit Cloud

---

## 🏁 Conclusion  

This project demonstrates how **Natural Language Processing and Machine Learning** can be used to build an **ATS-inspired resume screening system**.

By combining **semantic similarity**, **keyword coverage**, and **dynamic weighted scoring**, the system produces explainable and realistic match scores that closely reflect real-world hiring tools.

---

## 📌 Interview One-Liner  
I built an ATS-inspired resume analyzer using NLP techniques like TF-IDF, keyword overlap, and dynamic weighted scoring to realistically evaluate resume–job alignment.


