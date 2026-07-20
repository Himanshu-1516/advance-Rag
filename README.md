# 🧠 Advanced Graph RAG System

An advanced **Graph-based Retrieval-Augmented Generation (Graph RAG)** application that combines **Hybrid Search, Knowledge Graphs, Cross-Encoder Re-ranking, Semantic Caching, and Multi-Hop Retrieval** to deliver accurate, context-aware, and deterministic responses from PDF documents.

🔗 **Live Demo:** https://advance-rag15.streamlit.app/

---

## 🚀 Features

* **📄 PDF Question Answering**

  * Upload any PDF and chat with your document.

* **🔍 Hybrid Retrieval**

  * Combines **Dense Vector Search** and **BM25 Sparse Search** for improved retrieval accuracy.

* **🧠 Knowledge Graph Reasoning**

  * Automatically extracts relationships between entities and uses them to enhance context understanding.

* **🔄 Multi-Hop Query Decomposition**

  * Breaks complex questions into smaller sub-queries for better retrieval.

* **🎯 Cross-Encoder Re-ranking**

  * Re-ranks retrieved chunks to select the most relevant information before passing it to the LLM.

* **📚 Parent & Neighbor Chunk Expansion**

  * Expands retrieved chunks with neighboring context to avoid losing important information.

* **⚡ Semantic Cache**

  * Stores semantically similar queries to reduce latency and improve response speed.

* **📦 Context Compression**

  * Removes duplicate information and compresses retrieved context before generation.

* **🎛 Deterministic Mode**

  * Uses temperature = 0 to ensure consistent responses for identical queries.

* **🧪 Evaluation Harness**

  * Measure response consistency by running the same query multiple times.

* **💬 Multi-Chat Support**

  * Create and manage multiple document conversations independently.

---

## 🛠 Tech Stack

### AI & LLM

* Groq (Llama 3.1)
* LangChain
* Sentence Transformers
* Cross Encoder

### Retrieval

* Pinecone Vector Database
* FAISS
* BM25

### Knowledge Graph

* NetworkX

### Frontend

* Streamlit

### Language

* Python

---

## ⚙️ How It Works

1. Upload a PDF.
2. The document is split into overlapping chunks.
3. Dense embeddings and BM25 sparse vectors are created.
4. Chunks are indexed in Pinecone.
5. User queries are decomposed into multiple sub-queries.
6. Hybrid retrieval fetches the most relevant chunks.
7. A Cross-Encoder re-ranks the retrieved results.
8. Neighboring chunks are added to preserve context.
9. A Knowledge Graph extracts and connects entity relationships.
10. Context is compressed and cleaned.
11. The LLM generates an accurate, grounded response.
12. Semantic caching stores responses for faster future retrieval.

---

## 🎯 Key Highlights

* Hybrid Dense + Sparse Retrieval
* Knowledge Graph Enhanced RAG
* Cross-Encoder Re-ranking
* Multi-Hop Retrieval
* Context Compression
* Semantic Cache
* Deterministic Responses
* Evaluation Framework
* Production-style RAG Pipeline

---

## 📸 Live Demo

👉 **https://advance-rag15.streamlit.app/**

---

## 🎯 Future Improvements

* Multi-modal RAG (Text + Images)
* OCR support for scanned PDFs
* Agentic RAG workflows
* Citation-based answers
* Multi-document retrieval
* User authentication
* Cloud deployment with scalable infrastructure

---

## 🤝 Feedback

If you have suggestions, ideas, or feedback, feel free to open an issue or connect with me on LinkedIn.

If you found this project useful, consider giving it a ⭐ on GitHub!
