import streamlit as st
import os
import time
import tempfile
import traceback
import uuid
import numpy as np
import networkx as nx
import faiss
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
if "groq_key" not in st.session_state:
    st.session_state.groq_key = ""
if "pinecone_key" not in st.session_state:
    st.session_state.pinecone_key = ""

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
        self.cache_queries = []

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
            return self.cache_answers[indices[0][0]], float(distances[0][0])
        return None

    def add_to_cache(self, query, answer):
        vec = self._get_vector(query)
        self.index.add(vec)
        self.cache_answers.append(answer)
        self.cache_queries.append(query)


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
            except Exception:
                continue

    def get_graph_context(self, query):
        if self.graph.number_of_nodes() == 0:
            return ""
        try:
            entities = self.llm.invoke(
                f"Extract main entities/keywords from this query as comma-separated list: {query}"
            ).content.split(',')
            entities = [e.strip().lower() for e in entities if e.strip()]
        except Exception:
            return ""

        relations = []
        for ent in entities:
            for node in list(self.graph.nodes()):
                if ent in str(node).lower() or str(node).lower() in ent:
                    for u, v, data in self.graph.edges(node, data=True):
                        relations.append(f"• {u} → ({data.get('relation')}) → {v}")
        return "KNOWLEDGE GRAPH:\n" + "\n".join(list(set(relations))[:12]) if relations else ""


# Initialize Semantic Cache (do this safely, after faiss import is confirmed)
if "semantic_cache" not in st.session_state:
    st.session_state.semantic_cache = SemanticCache(embeddings)

# ========================= HELPER: WAIT FOR PINECONE INDEX =========================
def wait_for_index_ready(pc, index_name, timeout=60):
    start = time.time()
    while True:
        desc = pc.describe_index(index_name)
        status = desc.status if hasattr(desc, "status") else desc.get("status", {})
        ready = status.get("ready") if isinstance(status, dict) else getattr(status, "ready", False)
        if ready:
            return True
        if time.time() - start > timeout:
            raise TimeoutError(f"Pinecone index '{index_name}' did not become ready in time.")
        time.sleep(1)


# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("🛠️ Configuration")

    groq_key = st.text_input("Groq API Key", type="password", value=st.session_state.get("groq_key", ""))
    pinecone_key = st.text_input("Pinecone API Key", type="password", value=st.session_state.get("pinecone_key", ""))

    if groq_key:
        st.session_state.groq_key = groq_key
    if pinecone_key:
        st.session_state.pinecone_key = pinecone_key

    st.divider()
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    process_disabled = not (st.session_state.groq_key and st.session_state.pinecone_key and uploaded_file)

    if st.button("Process Document", type="primary", disabled=process_disabled):
        tmp_path = None
        try:
            with st.spinner("Processing PDF + Building Hybrid Index..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                # Load & split PDF
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                if not docs:
                    raise ValueError("No content could be extracted from the PDF.")

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)
                if not chunks:
                    raise ValueError("PDF produced no chunks after splitting.")

                texts = [chunk.page_content for chunk in chunks]

                # BM25
                st.write("🔧 Fitting BM25 encoder...")
                bm25 = BM25Encoder().default()
                bm25.fit(texts)
                st.session_state.bm25_encoder = bm25

                # Pinecone setup
                st.write("🔧 Connecting to Pinecone...")
                pc = Pinecone(api_key=st.session_state.pinecone_key)
                index_name = "graphrag"

                existing_indexes = [idx.name for idx in pc.list_indexes()]
                if index_name not in existing_indexes:
                    st.write("🔧 Creating new Pinecone index (this may take up to a minute)...")
                    pc.create_index(
                        name=index_name,
                        dimension=384,
                        metric="dotproduct",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                    wait_for_index_ready(pc, index_name, timeout=90)
                else:
                    # Even if it exists, make sure it's ready
                    wait_for_index_ready(pc, index_name, timeout=30)

                index = pc.Index(index_name)
                st.session_state.pinecone_index = index

                # Build & upsert vectors
                st.write("🔧 Embedding & upserting chunks into Pinecone...")
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

                # Upsert in batches to avoid payload size limits
                batch_size = 100
                for start_idx in range(0, len(vectors), batch_size):
                    batch = vectors[start_idx:start_idx + batch_size]
                    index.upsert(vectors=batch, namespace=st.session_state.session_id)

                st.session_state.pdf_processed = True
                st.success(f"✅ Document processed! {len(chunks)} chunks indexed.")

        except Exception as e:
            st.session_state.pdf_processed = False
            st.error(f"Error while processing document: {str(e)}")
            with st.expander("Full traceback (for debugging)"):
                st.code(traceback.format_exc())
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if st.session_state.pdf_processed:
            st.rerun()

    st.divider()
    if st.session_state.pdf_processed:
        st.success("📄 Document ready — you can chat now!")
    else:
        st.info("No document processed yet.")

    if st.button("Reset Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ========================= MAIN UI =========================
st.markdown('<p class="main-header">🧠 Advanced Graph RAG System</p>', unsafe_allow_html=True)
st.caption("HyDE + Hybrid Search + Cross-Encoder Reranking + Knowledge Graph + Semantic Cache")

if not st.session_state.pdf_processed:
    st.info("👈 Please add your API keys and upload a PDF in the sidebar, then click **Process Document** to start chatting.")
else:
    # Guard against missing resources (e.g. after partial state issues)
    if not st.session_state.bm25_encoder or not st.session_state.pinecone_index or not st.session_state.groq_key:
        st.error("Some required resources are missing. Please reprocess the document.")
        st.stop()

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

    query = st.chat_input("Ask any question about your document...")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            cache_hit = False
            hypothetical_doc = ""
            graph_context = ""
            final_context = ""
            response = ""

            with st.status("Thinking...", expanded=True) as status:
                # Check Cache
                try:
                    cache_result = st.session_state.semantic_cache.get_cached_answer(query)
                except Exception:
                    cache_result = None

                if cache_result:
                    answer, similarity = cache_result
                    cache_hit = True
                    status.update(label="Answered from Cache", state="complete")
                    response = answer
                else:
                    # HyDE
                    status.write("Generating Hypothetical Document (HyDE)...")
                    try:
                        hyde_prompt = f"""Write a detailed, factual, encyclopedic passage that would answer this question:
Question: {query}
Passage:"""
                        hypothetical_doc = llm.invoke(hyde_prompt).content
                    except Exception as e:
                        status.update(label="HyDE generation failed", state="error")
                        st.error(f"Error generating HyDE passage: {e}")
                        st.stop()

                    # Retrieve & Rerank
                    status.write("Retrieving & Reranking documents...")
                    try:
                        retrieved = retriever.invoke(hypothetical_doc)
                    except Exception as e:
                        status.update(label="Retrieval failed", state="error")
                        st.error(f"Error retrieving documents: {e}")
                        st.stop()

                    if not retrieved:
                        status.update(label="No documents found", state="error")
                        st.warning("No relevant documents were found for this query.")
                        st.stop()

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

                    status.update(label="Generating Final Answer...", state="running")

                    # Final Answer Generation
                    final_prompt = f"""You are an expert assistant. Answer the question **only** using the provided context and knowledge graph.
If you cannot answer properly, say so.

Question: {query}

Context:
{final_context}

Answer:"""

                    try:
                        response = llm.invoke(final_prompt).content
                    except Exception as e:
                        status.update(label="Answer generation failed", state="error")
                        st.error(f"Error generating final answer: {e}")
                        st.stop()

                    st.session_state.semantic_cache.add_to_cache(query, response)
                    status.update(label="Done", state="complete")

            # Display badge if cache hit
            if cache_hit:
                st.markdown(
                    f"<span class='badge cache-hit'>⚡ CACHE HIT ({similarity:.3f})</span>",
                    unsafe_allow_html=True
                )

            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            if not cache_hit:
                with st.expander("🔍 View Internal Process (For Portfolio)"):
                    tab1, tab2, tab3 = st.tabs(["HyDE Output", "Knowledge Graph", "Final Context Used"])
                    tab1.write(hypothetical_doc)
                    tab2.code(graph_context if graph_context else "No clear relationships extracted.")
                    tab3.write(final_context)
