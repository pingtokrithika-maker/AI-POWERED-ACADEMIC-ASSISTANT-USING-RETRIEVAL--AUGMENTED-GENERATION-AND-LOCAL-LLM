# 🤖 AI-Powered Academic Assistant (RAG + Local LLM)

## 📌 Overview

This project is an AI-powered academic assistant that uses **Retrieval-Augmented Generation (RAG)** with a **local LLM** to answer questions from documents.

It allows users to upload files and get accurate, context-based answers.

---

## 🚀 Features

* 📄 Document-based question answering
* 🔍 Semantic search using embeddings
* 🤖 Local LLM integration (Ollama)
* ⚡ Fast and efficient retrieval
* 🧠 Context-aware responses

---

## 🛠️ Tech Stack

* Python
* LangChain
* ChromaDB
* Ollama (LLM)

---

## 📂 Project Structure

```
├── app.py
├── ingest.py
├── rag_query.py
├── requirements.txt
├── data/
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run Ollama

```
ollama run llama2
```

### 3. Add documents

Place files inside `data/` folder

### 4. Ingest data

```
python ingest.py
```

### 5. Run project

```
python app.py
```

---



