from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

# Load environment variables (if needed)
load_dotenv()

# Get current directory and persistent Chroma path
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# ✅ Use free Hugging Face embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Load the existing Chroma vector store
db = Chroma(
    persist_directory=persistent_directory,  # correct param name
    embedding_function=embeddings
)

# User query
query = "Where does Gandalf meet Frodo?"

# ✅ Retrieve relevant documents based on query
# Using MMR to get diverse but relevant results
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,             # Number of final results (increased from 3)
        "fetch_k": 15,      # Number of candidates to consider (increased from 10)
        "lambda_mult": 0.3  # Lower value (0.3) favors more diverse results
    }
)

relevant_docs = retriever.invoke(query)

# ✅ Print relevant results
print("\nRelevant Documents:")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
