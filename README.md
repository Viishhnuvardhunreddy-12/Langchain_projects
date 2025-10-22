# 🧠 LangChain + HuggingFace + Chroma Vector DB — RAG Demo

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** pipeline using **LangChain**, **Hugging Face Embeddings**, and **Chroma Vector Database**.  

It loads text documents (like *The Lord of the Rings*), splits them into chunks, creates embeddings, stores them persistently in a Chroma vector database, and retrieves the most relevant chunks for a given user query.

---

## 🚀 Features

- ✅ **Document Loader** – Automatically loads `.txt`, `.pdf`, or `.md` files.  
- ✅ **Text Chunking** – Splits large text into manageable chunks for better embedding quality.  
- ✅ **Embeddings** – Uses the free model `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face.  
- ✅ **Persistent Vector Store** – Stores embeddings locally using **Chroma DB**.  
- ✅ **Query Retrieval** – Retrieves the most relevant chunks using **MMR (Maximal Marginal Relevance)** to ensure diversity in search results.  
- ✅ **Reusable Setup** – Once the database is created, you can directly query it anytime without rebuilding.

---

## 🏗️ Project Structure

```
project/
│
├── documents/
│   └── lord_of_the_rings.txt         # Example input document
│
├── db/
│   └── chroma_db/                    # Persistent Chroma database
│
├── create_vectorstore.py             # Builds and saves the vector DB
├── query_retriever.py                # Queries the vector DB
├── .env                              # (Optional) Environment variables
└── README.md                         # Project documentation
```

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Viishhnuvardhunreddy-12/Langchain_projects.git
   cd langchain-chroma-rag
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # (Linux/Mac)
   venv\Scripts\activate      # (Windows)
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Example `requirements.txt`:**
   ```txt
   langchain
   langchain-community
   chromadb
   sentence-transformers
   python-dotenv
   PyMuPDF
   ```

---

## 📂 Step 1 — Create and Persist the Vector Store

Run the following script once to create embeddings and save them in Chroma DB:

```bash
python create_vectorstore.py
```

This script will:
- Load the text from `documents/lord_of_the_rings.txt`
- Split it into chunks
- Create embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Save the vector store persistently in `db/chroma_db/`

If the database already exists, it will skip re-initialization.

---

## 🔍 Step 2 — Query the Vector Database

Once the Chroma DB is built, you can query it using:

```bash
python query_retriever.py
```

Example output:

```
Relevant Documents:
Document 1:
Gandalf met Frodo in Bag End to discuss the One Ring...
Source: Unknown
```

You can modify the query in `query_retriever.py`:
```python
query = "Where does Gandalf meet Frodo?"
```

---

## 🧩 Code Explanation

### 🗂️ `create_vectorstore.py`
- Loads text file from `/documents`
- Splits content into chunks of 1000 characters with 50 overlap
- Embeds text using Hugging Face embeddings
- Creates and persists Chroma vector DB

### 🔎 `query_retriever.py`
- Loads existing Chroma DB from `/db/chroma_db`
- Uses MMR retrieval to fetch diverse but relevant chunks
- Prints the most relevant document contents

---

## 🧠 Model Used

| Type | Model Name | Source |
|------|-------------|--------|
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face |

This model is **lightweight**, **fast**, and **free to use**, perfect for semantic similarity and retrieval tasks.

---

## 🧰 Environment Variables (Optional)

If you plan to add API keys or model configs, create a `.env` file in the root directory:

```
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
```

---

## 🧩 Future Enhancements

- Integrate **LLM response generation** (e.g., Gemini or OpenAI)
- Add **multi-file ingestion**
- Create a **Streamlit or Flask interface** for interactive querying
- Support **metadata filtering** for refined search

---

## 🧑‍💻 Author

**S. Viishhnu Vardhun Reddy**  
Data Analytics & AI Enthusiast  
📧 [Email me](mailto:viishhnureddy@gmail.com)  
🌐 [LinkedIn](https://www.linkedin.com/in/viishhnu-vardhun-reddy-syamalla/)  

---

## 🪪 License

This project is open-source under the **MIT License**.  
Feel free to modify and use it for learning or production.

---

> ⭐ *If you found this helpful, give the repo a star on GitHub!*
