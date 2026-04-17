from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Paths
VECTOR_PATH = "vector_store"

# Load embedding model
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load existing vector DB
vectordb = Chroma(
    persist_directory=VECTOR_PATH,
    embedding_function=embedding
)

# Create retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# Connect to Gemma
llm = Ollama(model="gemma3:1b")

# Ask user question
query = input("Ask a question: ")

# Retrieve relevant docs
docs = retriever.invoke(query)

# Combine context
context = "\n\n".join([doc.page_content for doc in docs])

# Create prompt
prompt = f"""
Answer the question ONLY using the context below.

Context:
{context}

Question:
{query}
"""

# Get response
response = llm.invoke(prompt)

print("\nAnswer:\n")
print(response)