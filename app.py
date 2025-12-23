import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re

# -------------------- STREAMLIT CONFIG --------------------
st.set_page_config(
    page_title="Resume Job Match Scorer",
    page_icon="📄",
    layout="wide"
)

# -------------------- UI HEADER --------------------
st.markdown(
    """
    <h2 style="color:#1f77b4;">📄 Resume–Job Match Analyzer</h2>
    <p>
    Upload your resume and paste a job description to see how well they match.<br>
    This tool uses <b>TF-IDF + Keyword Overlap + Weighted Scoring</b>.
    </p>
    """,
    unsafe_allow_html=True
)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("About")
    st.info(
        """
        - ATS-inspired Resume Analyzer  
        - Uses NLP & ML techniques  
        - Produces realistic match scores  
        """
    )

    st.header("How it works")
    st.write(
        """
        1. Upload resume (PDF)  
        2. Paste job description  
        3. Click Analyze  
        4. View match score  
        """
    )

# -------------------- STOPWORDS (LIGHTWEIGHT) --------------------
STOP_WORDS = {
    "the","is","and","to","of","in","for","on","with","as","by","at",
    "an","be","this","that","are","was","were","it","from","or","a"
}

# -------------------- HELPER FUNCTIONS --------------------

def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF resume"""
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
    """Basic text cleaning"""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text):
    """Remove stopwords using simple tokenization"""
    words = text.split()
    return " ".join(word for word in words if word not in STOP_WORDS)


def tfidf_similarity(resume_text, jd_text):
    """Compute TF-IDF cosine similarity"""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(
        tfidf_matrix[0:1], tfidf_matrix[1:2]
    )[0][0] * 100
    return round(score, 2)


def keyword_overlap_score(resume_text, jd_text):
    """ATS-style keyword overlap score"""
    resume_text = resume_text.lower()
    jd_text = jd_text.lower()

    jd_tokens = jd_text.split()

    jd_keywords = [
        w for w in jd_tokens
        if w.isalpha() and w not in STOP_WORDS and len(w) > 2
    ]

    jd_keywords = list(set(jd_keywords))
    matched = [kw for kw in jd_keywords if kw in resume_text]

    if not jd_keywords:
        return 0.0

    return round((len(matched) / len(jd_keywords)) * 100, 2)

# -------------------- MAIN APP --------------------

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

            # Preprocessing
            resume_processed = remove_stopwords(clean_text(resume_text))
            jd_processed = remove_stopwords(clean_text(job_description))

            # Scores
            tfidf_score = tfidf_similarity(
                resume_processed, jd_processed
            )

            overlap_score = keyword_overlap_score(
                resume_processed, jd_processed
            )

            jd_length = len(jd_processed.split())

            # Dynamic weighting
            if jd_length < 80:
                final_score = round(
                    0.3 * tfidf_score + 0.7 * overlap_score, 2
                )
            else:
                final_score = round(
                    0.6 * tfidf_score + 0.4 * overlap_score, 2
                )

            # -------------------- RESULTS --------------------
            st.subheader("📊 Results")

            st.metric("Final Match Score", f"{final_score}%")
            st.write(f"**TF-IDF Similarity:** {tfidf_score}%")
            st.write(f"**Keyword Overlap:** {overlap_score}%")

            # Visualization
            fig, ax = plt.subplots(figsize=(6, 0.6))
            colors = ["#ff4b4b", "#ffa726", "#0f9d58"]
            color_index = min(int(final_score // 33), 2)

            ax.barh([0], [final_score], color=colors[color_index])
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_xlabel("Match Percentage")
            ax.set_title("Resume–Job Match")
            st.pyplot(fig)

            # Feedback
            if final_score < 40:
                st.warning("Low match — consider tailoring your resume.")
            elif final_score < 70:
                st.info("Good match — your resume aligns fairly well.")
            else:
                st.success("Excellent match — strong alignment detected!")

# -------------------- ENTRY POINT --------------------

if __name__ == "__main__":
    main()
