import streamlit as st
import fitz
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="AI Smart Document Analyzer",
    page_icon="📄",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0b0f19;
}

.block-container {
    padding-top: 2rem;
}

.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #aab4c3;
    margin-bottom: 30px;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background: #151b29;
    border: 1px solid #293246;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.markdown(
    '<div class="title">📄 AI  Document Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a document and let AI/NLP analyze its content.'
    '</div>',
    unsafe_allow_html=True
)


# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

# Selected document type
if "selected_type" not in st.session_state:
    st.session_state["selected_type"] = None

# Selected AI feature
if "selected_feature" not in st.session_state:
    st.session_state["selected_feature"] = "summary"


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:

    st.header("⚙️ AI Document Analyzer")

    # -------------------------------------------------
    # SUPPORTED FORMATS
    # -------------------------------------------------

    st.write("### Supported Formats")

    if st.button(
        "📕 PDF",
        use_container_width=True
    ):
        st.session_state["selected_type"] = "pdf"

    if st.button(
        "📘 DOCX",
        use_container_width=True
    ):
        st.session_state["selected_type"] = "docx"

    if st.button(
        "📄 TXT",
        use_container_width=True
    ):
        st.session_state["selected_type"] = "txt"

    st.divider()

    # -------------------------------------------------
    # AI FEATURES
    # -------------------------------------------------

    st.write("### AI Features")

    if st.button(
        "📝 Text Summarization",
        use_container_width=True
    ):
        st.session_state["selected_feature"] = "summary"

    if st.button(
        "🔑 Keyword Extraction",
        use_container_width=True
    ):
        st.session_state["selected_feature"] = "keywords"

    if st.button(
        "📌 Important Points",
        use_container_width=True
    ):
        st.session_state["selected_feature"] = "points"

    if st.button(
        "📊 Document Statistics",
        use_container_width=True
    ):
        st.session_state["selected_feature"] = "statistics"

    if st.button(
        "📖 Full Text",
        use_container_width=True
    ):
        st.session_state["selected_feature"] = "fulltext"


# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------

file_types = {
    "pdf": ["pdf"],
    "docx": ["docx"],
    "txt": ["txt"]
}

selected = st.session_state["selected_type"]


if selected:

    uploaded_file = st.file_uploader(
        f"📤 Upload your {selected.upper()} document",
        type=file_types[selected],
        key=f"uploader_{selected}"
    )

else:

    uploaded_file = st.file_uploader(
        "📤 Upload your document",
        type=["pdf", "docx", "txt"],
        key="general_uploader"
    )


# -------------------------------------------------
# TEXT EXTRACTION
# -------------------------------------------------

def extract_pdf(file):

    text = ""

    pdf = fitz.open(
        stream=file.read(),
        filetype="pdf"
    )

    for page in pdf:
        text += page.get_text() + "\n"

    pdf.close()

    return text


def extract_docx(file):

    document = Document(file)

    text = ""

    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"

    return text


def extract_txt(file):

    return file.read().decode("utf-8")


# -------------------------------------------------
# CLEAN TEXT
# -------------------------------------------------

def clean_text(text):

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()


# -------------------------------------------------
# KEYWORD EXTRACTION
# -------------------------------------------------

def extract_keywords(text, number=10):

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=50
        )

        matrix = vectorizer.fit_transform([text])

        scores = matrix.toarray()[0]

        words = vectorizer.get_feature_names_out()

        keywords = sorted(
            zip(words, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            word
            for word, score in keywords[:number]
        ]

    except:

        return []


# -------------------------------------------------
# TEXT SUMMARIZATION
# -------------------------------------------------

def generate_summary(
    text,
    sentences_count=5
):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 30
    ]

    if len(sentences) <= sentences_count:

        return " ".join(sentences)

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        matrix = vectorizer.fit_transform(
            sentences
        )

        scores = matrix.sum(axis=1)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )

        selected = sorted(
            [
                index
                for index, score
                in ranked[:sentences_count]
            ]
        )

        summary = [
            sentences[index]
            for index in selected
        ]

        return " ".join(summary)

    except:

        return " ".join(
            sentences[:sentences_count]
        )


# -------------------------------------------------
# IMPORTANT POINTS
# -------------------------------------------------

def important_points(text):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 40
    ]

    if len(sentences) > 5:

        return sentences[:5]

    return sentences


# -------------------------------------------------
# PROCESS DOCUMENT
# -------------------------------------------------

if uploaded_file:

    file_name = uploaded_file.name

    # -------------------------------------------------
    # DETERMINE FILE TYPE
    # -------------------------------------------------

    if file_name.lower().endswith(".pdf"):

        text = extract_pdf(uploaded_file)

    elif file_name.lower().endswith(".docx"):

        text = extract_docx(uploaded_file)

    elif file_name.lower().endswith(".txt"):

        text = extract_txt(uploaded_file)

    else:

        text = ""

    # -------------------------------------------------
    # CLEAN TEXT
    # -------------------------------------------------

    text = clean_text(text)

    # -------------------------------------------------
    # SUCCESS MESSAGE
    # -------------------------------------------------

    st.success(
        f"✅ {file_name} uploaded successfully!"
    )

    # -------------------------------------------------
    # IF TEXT EXISTS
    # -------------------------------------------------

    if text:

        # -------------------------------------------------
        # CALCULATE DOCUMENT DATA
        # -------------------------------------------------

        words = text.split()

        characters = len(text)

        sentences = re.split(
            r'(?<=[.!?])\s+',
            text
        )

        # -------------------------------------------------
        # AI DOCUMENT ANALYSIS
        # -------------------------------------------------

        st.header(
            "🤖 AI Document Analysis"
        )

        selected_feature = (
            st.session_state["selected_feature"]
        )


        # =================================================
        # SUMMARY
        # =================================================

        if selected_feature == "summary":

            st.subheader(
                "📝 Automatic Summary"
            )

            summary = generate_summary(text)

            st.markdown(
                f"""
                <div class="card">
                {summary}
                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # KEYWORDS
        # =================================================

        elif selected_feature == "keywords":

            st.subheader(
                "🔑 Important Keywords"
            )

            keywords = extract_keywords(text)

            if keywords:

                for keyword in keywords:

                    st.write(
                        f"🔹 **{keyword}**"
                    )

            else:

                st.warning(
                    "Could not extract keywords."
                )


        # =================================================
        # IMPORTANT POINTS
        # =================================================

        elif selected_feature == "points":

            st.subheader(
                "📌 Important Points"
            )

            points = important_points(text)

            if points:

                for point in points:

                    st.write(
                        f"• {point}"
                    )

            else:

                st.warning(
                    "No important points found."
                )


        # =================================================
        # DOCUMENT STATISTICS
        # =================================================

        elif selected_feature == "statistics":

            st.subheader(
                "📊 Document Statistics"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "📝 Words",
                    len(words)
                )

            with col2:

                st.metric(
                    "🔤 Characters",
                    characters
                )

            with col3:

                st.metric(
                    "📄 Sentences",
                    len(sentences)
                )

            with col4:

                st.metric(
                    "📦 File Type",
                    file_name
                    .split(".")[-1]
                    .upper()
                )


        # =================================================
        # FULL TEXT
        # =================================================

        elif selected_feature == "fulltext":

            st.subheader(
                "📖 Extracted Document Text"
            )

            st.text_area(
                "Document Content",
                text,
                height=450
            )


        # -------------------------------------------------
        # DOWNLOAD REPORT
        # -------------------------------------------------

        summary = generate_summary(text)

        keywords = extract_keywords(text)

        points = important_points(text)

        report = f"""
AI SMART DOCUMENT ANALYZER
===========================

Document:
{file_name}

DOCUMENT STATISTICS
-------------------
Words: {len(words)}
Characters: {characters}
Sentences: {len(sentences)}
File Type: {file_name.split(".")[-1].upper()}

SUMMARY
-------
{summary}

KEYWORDS
--------
{", ".join(keywords)}

IMPORTANT POINTS
----------------
"""

        for point in points:

            report += f"\n• {point}"

        report += (
            "\n\nFULL DOCUMENT TEXT\n"
            "------------------\n"
        )

        report += text

        st.divider()

        st.subheader(
            "📥 Download Analysis"
        )

        st.download_button(
            label="⬇️ Download Analysis Report",
            data=report,
            file_name="document_analysis.txt",
            mime="text/plain"
        )


    # -------------------------------------------------
    # NO TEXT FOUND
    # -------------------------------------------------

    else:

        st.warning(
            "⚠️ No readable text was found "
            "in this document."
        )


# -------------------------------------------------
# NO DOCUMENT UPLOADED
# -------------------------------------------------

else:

    st.info(
        "👆 Upload a PDF, DOCX, or TXT document "
        "to begin analysis."
    )
