import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(
    page_title="Resume Job Match Scorer",
    page_icon="📄",
    layout="wide"
)

st.markdown(
    """
    <h2 style="color:red;">📄 Resume–Job Match Analyzer</h2>
    <p style="color:red;">
    Upload your resume and paste a job description to see how well they match.<br>
    This tool uses <b>TF-IDF + Keyword Overlap + Weighted Scoring</b>.
    </p>
    """,
    unsafe_allow_html=True
)



with st.sidebar:
    st.header("About")
    st.info("""
    - Resume vs Job Description matching
    - Uses NLP & cosine similarity
    - Provides explainable match score
    """)

    st.header("How it works")
    st.write("""
    1. Upload resume (PDF)  
    2. Paste job description  
    3. Click **Analyze Match**  
    4. View score & insights  
    """)


def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return " ".join([word for word in words if word not in stop_words])


def tfidf_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(
        tfidf_matrix[0:1], tfidf_matrix[1:2]
    )[0][0] * 100
    return round(score, 2)


def keyword_overlap_score(resume_text, jd_text):
    resume_text = resume_text.lower()
    jd_text = jd_text.lower()

    stop_words = set(stopwords.words('english'))

    jd_tokens = word_tokenize(jd_text)
    jd_keywords = [
        w for w in jd_tokens
        if w.isalpha() and w not in stop_words and len(w) > 2
    ]

    jd_keywords = list(set(jd_keywords))  

    matched = [kw for kw in jd_keywords if kw in resume_text]

    if not jd_keywords:
        return 0.0

    return round((len(matched) / len(jd_keywords)) * 100, 2)



def main():
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF)", type=["pdf"]
    )
    job_description = st.text_area(
        "Paste the job description", height=200
    )

    if st.button("Analyze Match"):
        if not uploaded_file:
            st.warning("Please upload your resume.")
            return

        if not job_description.strip():
            st.warning("Please paste the job description.")
            return

        with st.spinner("Analyzing your resume..."):
            resume_text = extract_text_from_pdf(uploaded_file)

            if not resume_text.strip():
                st.error("Could not extract text from the PDF.")
                return

            resume_processed = remove_stopwords(clean_text(resume_text))
            jd_processed = remove_stopwords(clean_text(job_description))

            tfidf_score = tfidf_similarity(
                resume_processed, jd_processed
            )

            overlap_score = keyword_overlap_score(
                resume_processed, jd_processed
            )
            jd_length = len(jd_processed.split())

            if jd_length < 80:   
                final_score = round(
                    0.4 * tfidf_score + 0.6* overlap_score, 2
                    )
            else:              
                final_score = round(
                    0.2 * tfidf_score + 0.8 * overlap_score, 2
                    )
           

            st.subheader("📊 Results")

            st.metric("Final Match Score", f"{final_score}%")
            st.write(f"**TF-IDF Similarity:** {tfidf_score}%")
            st.write(f"**Keyword Overlap:** {overlap_score}%")

            # Gauge-style bar
            fig, ax = plt.subplots(figsize=(6, 0.6))
            colors = ['#ff4b4b', '#ffa726', '#0f9d58']
            color_index = min(int(final_score // 33), 2)

            ax.barh([0], [final_score], color=colors[color_index])
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_xlabel("Match Percentage")
            ax.set_title("Resume–Job Match")
            st.pyplot(fig)

            if final_score < 40:
                st.warning("Low match — consider tailoring your resume.")
            elif final_score < 70:
                st.info("Good match — your resume aligns fairly well.")
            else:
                st.success("Excellent match — strong alignment detected!")


if __name__ == "__main__":
    main()


