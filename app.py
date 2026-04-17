import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from duckduckgo_search import DDGS

# -----------------------
# Initialize Model
# -----------------------
llm = Ollama(model="gemma:2b")

st.title("📚 AI Academic Assistant")

# -----------------------
# Session State
# -----------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "equations" not in st.session_state:
    st.session_state.equations = ""

if "quiz" not in st.session_state:
    st.session_state.quiz = ""

# -----------------------
# Upload PDF
# -----------------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Full text
    full_text = " ".join([doc.page_content for doc in documents])

    # Split + embeddings
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    # -----------------------
    # DOCUMENT ANALYSIS
    # -----------------------
    st.subheader("📊 Document Analysis")

    col1, col2, col3 = st.columns(3)

    # SUMMARY
    with col1:
        if st.button("📖 Summarize"):
            st.session_state.summary = llm.invoke(
                "Summarize this document in bullet points:\n" + full_text
            )

    # EQUATIONS
    with col2:
        if st.button("📐 Explain Equations"):
            st.session_state.equations = llm.invoke(
                "Extract and explain all formulas clearly:\n" + full_text
            )

    # QUIZ (FIXED FORMAT)
    with col3:
        if st.button("📝 Generate Quiz"):
            st.session_state.quiz = llm.invoke(
                """
Generate 5 multiple choice questions strictly in this format:

1. Question
A. Option A
B. Option B
C. Option C
D. Option D
Answer: Correct option

Make sure:
- Use A, B, C, D format only
- Each question is clearly separated
- Proper line spacing

Content:
""" + full_text
            )

    # -----------------------
    # DISPLAY OUTPUTS
    # -----------------------
    if st.session_state.summary:
        st.markdown("## 📖 Summary")
        st.write(st.session_state.summary)

    if st.session_state.equations:
        st.markdown("## 📐 Equations")
        st.write(st.session_state.equations)

    if st.session_state.quiz:
        st.markdown("## 📝 Quiz")

        quiz_text = st.session_state.quiz

        # Formatting for clean display
        formatted_quiz = quiz_text.replace("A.", "\nA.") \
                                  .replace("B.", "\nB.") \
                                  .replace("C.", "\nC.") \
                                  .replace("D.", "\nD.") \
                                  .replace("Answer:", "\n\nAnswer:")

        st.text(formatted_quiz)

    # -----------------------
    # QUESTION INPUT (RAG)
    # -----------------------
    st.subheader("💬 Ask Question")
    query = st.text_input("Enter your question")

    if query:
        st.session_state.chat_history.append(query)

        docs = retriever.invoke(query)

        # INTERNET FALLBACK
        if len(docs) == 0:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=2))
                web_data = " ".join([r["body"] for r in results])

            answer = llm.invoke(
                f"Answer using this web info:\n{web_data}\nQuestion:{query}"
            )

            st.write("🌐 Internet Answer:")
            st.write(answer)

        else:
            context = ""
            sources = []

            for doc in docs:
                context += doc.page_content
                if "page" in doc.metadata:
                    sources.append(f"Page {doc.metadata['page']}")

            history = " ".join(st.session_state.chat_history)

            answer = llm.invoke(
                f"History:{history}\nContext:{context}\nQuestion:{query}"
            )

            st.write("### ✅ Answer:")
            st.write(answer)

            st.write("### 📌 Source Pages:")
            st.write(", ".join(sources))
