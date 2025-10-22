import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader,PyPDFLoader,PyMuPDFLoader
from langchain_community.vectorstores import Chroma

# Path settings (your document and Chroma DB)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Use the actual filename present in the documents directory
file_path = os.path.join(current_dir, "documents", "lord_of_the_rings.txt")  # Ensure it's .txt
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Check whether Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing the vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # Load the text content from the file
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split the loaded documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # ✅ Create embeddings using a free Hugging Face model
    print("Creating embeddings using Hugging Face model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Finished creating the embeddings!")

    # ✅ Create the vector store and persist it
    db = Chroma.from_documents(
        docs,
        embedding=embeddings,               # correct argument name
        persist_directory=persistent_directory
    )
    print("Finished creating and saving the vector store!")

else:
    print("Vector store already created. No need to initialize again.")
