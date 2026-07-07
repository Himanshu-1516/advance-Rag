
import streamlit as st
import os
import tempfile
import uuid
import numpy as np
import networkx as nx
import faiss  # <-- FIXED: Added missing faiss import
from sentence_transformers import CrossEncoder

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone, ServerlessSpec
from langchain_community.retrievers import PineconeHybridSearchRetriever

# ========================= PAGE CONFIG =========================
st.set_page_config(page_title="Graph RAG • Live Demo", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 42px; font-weight: bold; color: #1E3A8A;}
    .badge {padding: 4px 12px; border-radius: 12px; font-size: 13px; font-weight: bold;}
    .cache-hit {background-color: #22c55e; color: white;}
    .status-box {padding: 15px; border-radius: 10px; background-color: #f8fafc; border: 1px solid #e2e8f0;}
</style>
""", unsafe_allow_html=True)

# ========================= SESSION STATE =========================
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "bm25_encoder" not in st.session_state:
    st.session_state.bm25_encoder = None
if "pinecone_index" not in st.session_state:
    st.session_state.pinecone_index = None

# ========================= CACHED RESOURCES =========================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_reranker():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

embeddings = load_embeddings()
reranker = load_reranker()

# ========================= CLASSES =========================
class SemanticCache:
    def __init__(self, embeddings_model, threshold=0.82):
        self.embeddings = embeddings_model
        self.threshold = threshold
        self.dim = 384
        self.index = faiss.IndexFlatIP(self.dim)
        self.cache_answers = []

    def _get_vector(self, text):
        vec = np.array([self.embeddings.embed_query(text)], dtype=np.float32)
        faiss.normalize_L2(vec)
        return vec

    def get_cached_answer(self, query):
        if self.index.ntotal == 0:
            return None
        vec = self._get_vector(query)
        distances, indices = self.index.search(vec, 1)
        if distances[0][0] >= self.threshold:
            return self.cache_answers[indices[0][0]], distances[0][0]
        return None

    def add_to_cache(self, query, answer):
        vec = self._get_vector(query)
        self.index.add(vec)
        self.cache_answers.append(answer)

class KnowledgeGraphRAG:
    def __init__(self, llm):
        self.llm = llm
        self.graph = nx.Graph()

    def build_graph(self, documents):
        prompt = """Extract only factual knowledge triplets from the text.
        Output format: Entity1 | Relationship | Entity2
        Do not hallucinate. If nothing clear, output 'NONE'.
        Text: {text}"""
        
        for doc in documents:
            try:
                res = self.llm.invoke(prompt.format(text=doc.page_content)).content
                for line in res.splitlines():
                    if '|' in line:
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) == 3:
                            self.graph.add_edge(parts[0], parts[2], relation=parts[1])
            except:
                continue

    def get_graph_context(self, query):
        if self.graph.number_of_nodes() == 0:
            return ""
        try:
            entities = self.llm.invoke(
                f"Extract main entities/keywords from this query as comma-separated list: {query}"
            ).content.split(',')
            entities = [e.strip().lower() for e in entities if e.strip()]
        except:
            return ""

        relations = []
        for ent in entities:
            for node in list(self.graph.nodes()):
                if ent in str(node).lower() or str(node).lower() in ent:
                    for u, v, data in self.graph.edges(node, data=True):
                        relations.append(f"• {u} → ({data.get('relation')}) → {v}")
        return "KNOWLEDGE GRAPH:\n" + "\n".join(list(set(relations))[:12]) if relations else ""

# Initialize Semantic Cache
if "semantic_cache" not in st.session_state:
    st.session_state.semantic_cache = SemanticCache(embeddings)

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("🛠️ Configuration")
    
    groq_key = st.text_input("Groq API Key", type="password", value=st.session_state.get("groq_key", ""))
    pinecone_key = st.text_input("Pinecone API Key", type="password", value=st.session_state.get("pinecone_key", ""))
    
    if groq_key and pinecone_key:
        st.session_state.groq_key = groq_key
        st.session_state.pinecone_key = pinecone_key

    st.divider()
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if st.button("Process Document", type="primary", disabled=not (groq_key and pinecone_key and uploaded_file)):
        with st.spinner("Processing PDF + Building Hybrid Index + Knowledge Graph..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)

                texts = [chunk.page_content for chunk in chunks]

                # BM25
                bm25 = BM25Encoder().default()
                bm25.fit(texts)
                st.session_state.bm25_encoder = bm25

                # Pinecone
                pc = Pinecone(api_key=pinecone_key)
                index_name = "graphrag"
                if index_name not in [idx.name for idx in pc.list_indexes()]:
                    pc.create_index(
                        name=index_name,
                        dimension=384,
                        metric="dotproduct",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                index = pc.Index(index_name)
                st.session_state.pinecone_index = index

                # Upsert with namespace (user isolation)
                vectors = []
                for i, (text, chunk) in enumerate(zip(texts, chunks)):
                    dense = embeddings.embed_query(text)
                    sparse = bm25.encode_documents([text])[0]
                    vectors.append({
                        "id": f"chunk_{i}",
                        "values": dense,
                        "sparse_values": sparse,
                        "metadata": {
                            "context": text,
                            "page": chunk.metadata.get("page", 0),
                            "source": uploaded_file.name
                        }
                    })

                index.upsert(vectors=vectors, namespace=st.session_state.session_id)
                st.session_state.pdf_processed = True
                os.unlink(tmp_path)
                st.success("✅ Document successfully processed! You can now start chatting.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if st.button("Reset Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ========================= MAIN UI =========================
st.markdown('<p class="main-header">🧠 Advanced Graph RAG System</p>', unsafe_allow_html=True)
st.caption("HyDE + Hybrid Search + Cross-Encoder Reranking + Knowledge Graph + Semantic Cache")

if not st.session_state.pdf_processed:
    st.info("👈 Please add your API keys and upload a PDF in the sidebar, then click 'Process Document' to start chatting.")
else:
    llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=st.session_state.groq_key)
    kg_rag = KnowledgeGraphRAG(llm)

    # Retriever
    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=st.session_state.bm25_encoder,
        index=st.session_state.pinecone_index,
        alpha=0.5,
        top_k=8,
        namespace=st.session_state.session_id
    )

    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # CHAT INPUT IS HERE (Only appears after PDF is processed)
    query = st.chat_input("Ask any question about your document...")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.status("Thinking...", expanded=True) as status:
                # Check Cache
                cache_result = st.session_state.semantic_cache.get_cached_answer(query)
                if cache_result:
                    answer, similarity = cache_result
                    st.markdown(f"<span class='badge cache-hit'>⚡ CACHE HIT ({similarity:.3f})</span>", unsafe_allow_html=True)
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    status.update(label="Answered from Cache", state="complete")
                    st.stop()

                # HyDE
                status.write("Generating Hypothetical Document (HyDE)...")
                hyde_prompt = f"""Write a detailed, factual, encyclopedic passage that would answer this question:
                Question: {query}
                Passage:"""
                hypothetical_doc = llm.invoke(hyde_prompt).content

                # Retrieve & Rerank
                status.write("Retrieving & Reranking documents...")
                retrieved = retriever.invoke(hypothetical_doc)
                doc_texts = [doc.page_content for doc in retrieved]
                scores = reranker.predict([[hypothetical_doc, text] for text in doc_texts])
                ranked = sorted(zip(scores, retrieved), key=lambda x: x[0], reverse=True)
                top_docs = [doc for _, doc in ranked[:5]]

                # Knowledge Graph
                status.write("Building Knowledge Graph...")
                kg_rag.build_graph(top_docs)
                graph_context = kg_rag.get_graph_context(query)

                # Construct final context
                context_parts = [f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}" for doc in top_docs]
                final_context = "\n\n---\n\n".join(context_parts)
                if graph_context:
                    final_context = graph_context + "\n\n" + final_context

                status.update(label="Generating Final Answer", state="running")

            # Final Answer Generation
            final_prompt = f"""You are an expert assistant. Answer the question **only** using the provided context and knowledge graph.
If you cannot answer properly, say so.

Question: {query}

Context:
{final_context}

Answer:"""

            response = llm.invoke(final_prompt).content
            
            st.markdown(response)
            st.session_state.semantic_cache.add_to_cache(query, response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            with st.expander("🔍 View Internal Process (For Portfolio)"):
                tab1, tab2, tab3 = st.tabs(["HyDE Output", "Knowledge Graph", "Final Context Used"])
                tab1.write(hypothetical_doc)
                tab2.code(graph_context if graph_context else "No clear relationships extracted.")
                tab3.write(final_context)

