
import streamlit as st
import os
import time
import tempfile
import uuid
import re
import numpy as np
import networkx as nx
import faiss
import nltk
from sentence_transformers import CrossEncoder

# Download NLTK data required for BM25 (prevents blank screen/silent crashes)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone, ServerlessSpec
from langchain_community.retrievers import PineconeHybridSearchRetriever

# ========================= API KEYS =========================
# 🔑 Put your own API keys here (or set them as environment variables /
# Streamlit secrets with the same names). End users of your deployed app
# will NOT need to provide any keys themselves.

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "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")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY", "pcsk_39EGLB_PC9i9y7MQo2FxSqgqdX4akFP3LPFoNqHirwHsicYqAivgQASB4bFsM9ocPY9epZ")
GROQ_MODEL = os.getenv("GROQ_MODEL") or st.secrets.get("GROQ_MODEL", "llama-3.1-8b-instant")

KEYS_CONFIGURED = (
    GROQ_API_KEY and "PASTE_YOUR" not in GROQ_API_KEY and
    PINECONE_API_KEY and "PASTE_YOUR" not in PINECONE_API_KEY
)

# ========================= PAGE CONFIG =========================
st.set_page_config(page_title="Graph RAG • Live Demo", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 42px; font-weight: bold; color: #1E3A8A;}
    .badge {padding: 4px 12px; border-radius: 12px; font-size: 13px; font-weight: bold;}
    .cache-hit {background-color: #22c55e; color: white;}
    .status-box {padding: 15px; border-radius: 10px; background-color: #f8fafc; border: 1px solid #e2e8f0;}
    .chat-item-active {background-color: #1E3A8A !important; color: white !important;}
</style>
""", unsafe_allow_html=True)

if not KEYS_CONFIGURED:
    st.error(
        "⚠️ API keys are not configured yet. The app owner needs to set "
        "`GROQ_API_KEY` and `PINECONE_API_KEY` at the top of the script "
        "(or via environment variables / `st.secrets`) before this app can be used."
    )
    st.stop()

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
    """
    NEW AND IMPROVED KNOWLEDGE GRAPH:
    Extracts triplets reliably and uses keyword scoring instead of brittle LLM extraction 
    to guarantee information is successfully retrieved from the graph.
    """
    def __init__(self, llm):
        self.llm = llm
        self.graph = nx.Graph()

    def build_graph(self, documents):
        prompt = """Extract the most important factual relationships from the text.
Strict Rules:
1. Output ONLY triplets in this exact format: Entity1 | Relationship | Entity2
2. Do not use markdown, do not use numbers, do not add any conversational text.

Text: {text}"""

        for doc in documents:
            try:
                res = self.llm.invoke(prompt.format(text=doc.page_content)).content
                for line in res.splitlines():
                    # Only process lines that actually look like our requested triplet
                    if line.count('|') == 2:
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) == 3 and parts[0].lower() != "none":
                            self.graph.add_edge(parts[0], parts[2], relation=parts[1])
            except Exception:
                continue

    def get_graph_context(self, query):
        if self.graph.number_of_nodes() == 0:
            return ""
        
        # 1. Get words from the query (ignore tiny words like "is", "a", "the")
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        query_words = {w for w in query_words if len(w) > 3}

        all_edges = []
        for u, v, data in self.graph.edges(data=True):
            all_edges.append((u, v, data.get('relation', '')))

        # 2. Score each relationship based on how much it overlaps with the user's query
        def edge_score(edge):
            u, v, rel = edge
            text = f"{u} {v} {rel}".lower()
            return sum(1 for w in query_words if w in text)

        # Sort edges: highest relevance to query first
        scored_edges = sorted(all_edges, key=edge_score, reverse=True)

        # 3. Format top 20 most relevant facts to inject into prompt
        relations = []
        for u, v, rel in scored_edges[:20]:
            relations.append(f"• {u} → ({rel}) → {v}")

        if not relations:
            return ""

        return "EXTRACTED KNOWLEDGE GRAPH FACTS:\n" + "\n".join(relations)

# ========================= MULTI-CHAT SESSION STATE =========================
def create_new_chat(name=None):
    """Create a brand-new, isolated chat session and return its id."""
    chat_id = f"chat_{uuid.uuid4().hex[:8]}"
    st.session_state.chats[chat_id] = {
        "name": name or f"Chat {len(st.session_state.chats) + 1}",
        "chat_history": [],
        "pdf_processed": False,
        "bm25_encoder": None,
        "pinecone_index": None,
        "namespace": chat_id,          # isolates vectors per chat inside Pinecone
        "semantic_cache": SemanticCache(embeddings),
        "doc_name": None,
    }
    return chat_id


if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chats:
    st.session_state.current_chat_id = create_new_chat("Chat 1")

# ========================= HELPER =========================
def wait_for_index_ready(pc, index_name, timeout=90):
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

# ========================= SIDEBAR: CHAT MANAGEMENT =========================
with st.sidebar:
    st.header("💬 Chats")

    if st.button("➕ New Chat", use_container_width=True):
        new_id = create_new_chat()
        st.session_state.current_chat_id = new_id
        st.rerun()

    st.divider()

    for cid, cdata in list(st.session_state.chats.items()):
        col1, col2 = st.columns([5, 1])
        with col1:
            label = ("📄 " if cdata["pdf_processed"] else "🗒️ ") + cdata["name"]
            btn_type = "primary" if cid == st.session_state.current_chat_id else "secondary"
            if st.button(label, key=f"select_{cid}", use_container_width=True, type=btn_type):
                st.session_state.current_chat_id = cid
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{cid}"):
                del st.session_state.chats[cid]
                if not st.session_state.chats:
                    create_new_chat("Chat 1")
                if st.session_state.current_chat_id == cid:
                    st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
                st.rerun()

    st.divider()
    st.header("🛠️ Document Setup")

    chat = st.session_state.chats[st.session_state.current_chat_id]

    new_name = st.text_input("Chat name", value=chat["name"], key=f"name_{st.session_state.current_chat_id}")
    if new_name and new_name != chat["name"]:
        chat["name"] = new_name

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type="pdf",
        key=f"upload_{st.session_state.current_chat_id}"  # unique per chat -> no leakage between chats
    )

    process_disabled = uploaded_file is None

    if st.button("Process Document", type="primary", disabled=process_disabled):
        tmp_path = None
        try:
            with st.spinner("Processing PDF + Building Hybrid Index..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                if not docs:
                    raise ValueError("No content could be extracted from the PDF.")

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)

                texts = [chunk.page_content for chunk in chunks]

                st.write("🔧 Fitting BM25 encoder...")
                bm25 = BM25Encoder().default()
                bm25.fit(texts)
                chat["bm25_encoder"] = bm25

                st.write("🔧 Connecting to Pinecone...")
                pc = Pinecone(api_key=PINECONE_API_KEY)
                index_name = "graphrag"

                existing_indexes = [idx.name for idx in pc.list_indexes()]
                if index_name not in existing_indexes:
                    st.write("🔧 Creating new Pinecone index (may take up to a minute)...")
                    pc.create_index(
                        name=index_name,
                        dimension=384,
                        metric="dotproduct",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                    wait_for_index_ready(pc, index_name, timeout=90)
                else:
                    wait_for_index_ready(pc, index_name, timeout=30)

                index = pc.Index(index_name)
                chat["pinecone_index"] = index

                st.write("🔧 Embedding & upserting chunks into Pinecone...")
                vectors = []
                for i, (text, chunk_) in enumerate(zip(texts, chunks)):
                    dense = embeddings.embed_query(text)
                    sparse = bm25.encode_documents([text])[0]
                    vectors.append({
                        "id": f"chunk_{i}",
                        "values": dense,
                        "sparse_values": sparse,
                        "metadata": {
                            "context": text,
                            "page": chunk_.metadata.get("page", 0),
                            "source": uploaded_file.name
                        }
                    })

                batch_size = 100
                for start_idx in range(0, len(vectors), batch_size):
                    batch = vectors[start_idx:start_idx + batch_size]
                    index.upsert(vectors=batch, namespace=chat["namespace"])

                chat["pdf_processed"] = True
                chat["doc_name"] = uploaded_file.name
                if chat["name"].startswith("Chat "):
                    chat["name"] = uploaded_file.name.rsplit(".", 1)[0][:30]

                st.success(f"✅ Document processed! {len(chunks)} chunks indexed.")

        except Exception as e:
            chat["pdf_processed"] = False
            st.error(f"Error while processing document: {str(e)}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if chat["pdf_processed"]:
            st.rerun()

    st.divider()
    if st.button("Reset Current Chat"):
        cid = st.session_state.current_chat_id
        name = st.session_state.chats[cid]["name"]
        del st.session_state.chats[cid]
        st.session_state.current_chat_id = create_new_chat(name)
        st.rerun()

    if st.button("Clear All Chats"):
        st.session_state.chats = {}
        st.session_state.current_chat_id = create_new_chat("Chat 1")
        st.rerun()

# ========================= MAIN UI =========================
st.markdown('<p class="main-header">🧠 Advanced Graph RAG System</p>', unsafe_allow_html=True)
st.caption("HyDE + Hybrid Search + Cross-Encoder Reranking + Knowledge Graph + Semantic Cache")

chat = st.session_state.chats[st.session_state.current_chat_id]
st.subheader(f"💬 {chat['name']}" + (f"  ·  📄 {chat['doc_name']}" if chat["doc_name"] else ""))

if not chat["pdf_processed"]:
    st.info("👈 Upload a PDF for this chat in the sidebar, then click **Process Document** to start chatting.")
    st.stop()

# Display chat history
for message in chat["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask any question about your document...")

if query:
    chat["chat_history"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        clean_query = ''.join(c for c in query.lower() if c.isalnum() or c.isspace()).strip()
        conversational_greetings = ["hi", "hello", "hey", "good morning", "good evening", "how are you", "who are you", "sup"]

        if clean_query in conversational_greetings:
            response = "Hello! 👋 I am your Advanced Graph RAG Assistant. I've processed your document and built a knowledge graph. What would you like to know about it?"
            st.markdown(response)
            chat["chat_history"].append({"role": "assistant", "content": response})
        else:
            cache_hit = False
            similarity = 0.0
            hypothetical_doc = ""
            graph_context = ""
            final_context = ""
            response = ""

            try:
                if not chat["bm25_encoder"] or not chat["pinecone_index"]:
                    st.error("Required resources missing. Please reprocess the document.")
                    st.stop()

                llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)

                retriever = PineconeHybridSearchRetriever(
                    embeddings=embeddings,
                    sparse_encoder=chat["bm25_encoder"],
                    index=chat["pinecone_index"],
                    alpha=0.5,
                    top_k=8,
                    namespace=chat["namespace"]
                )

                kg_rag = KnowledgeGraphRAG(llm)

                with st.status("Thinking...", expanded=True) as status:
                    try:
                        cache_result = chat["semantic_cache"].get_cached_answer(query)
                    except Exception:
                        cache_result = None

                    if cache_result:
                        response, similarity = cache_result
                        cache_hit = True
                        status.update(label="Answered from Cache", state="complete")
                    else:
                        status.write("Generating Hypothetical Document (HyDE)...")
                        hyde_prompt = f"""Write a detailed, factual passage that answers this question:
Question: {query}
Passage:"""
                        hypothetical_doc = llm.invoke(hyde_prompt).content

                        status.write("Retrieving & Reranking documents...")
                        retrieved = retriever.invoke(hypothetical_doc)

                        if retrieved:
                            doc_texts = [doc.page_content for doc in retrieved]
                            scores = reranker.predict([[hypothetical_doc, text] for text in doc_texts])
                            ranked = sorted(zip(scores, retrieved), key=lambda x: x[0], reverse=True)
                            top_docs = [doc for _, doc in ranked[:5]]

                            status.write("Extracting Knowledge Graph Facts...")
                            kg_rag.build_graph(top_docs)
                            graph_context = kg_rag.get_graph_context(query)

                            context_parts = [f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}" for doc in top_docs]
                            final_context = "\n\n---\n\n".join(context_parts)
                            if graph_context:
                                final_context = graph_context + "\n\n---\nDOCUMENT TEXT:\n" + final_context
                        else:
                            final_context = "No relevant context found in the document."

                        status.update(label="Generating Final Answer...", state="running")

                        final_prompt = f"""You are an expert AI assistant. 
1. Answer the question using ONLY the provided Knowledge Graph Facts and Document Text.
2. If the context does not contain the answer, say: "I don't have enough information in the document to answer that." Do not hallucinate.

Question: {query}

Context Data:
{final_context}

Answer:"""
                        response = llm.invoke(final_prompt).content
                        chat["semantic_cache"].add_to_cache(query, response)
                        status.update(label="Done", state="complete")

                if cache_hit:
                    st.markdown(
                        f"<span class='badge cache-hit'>⚡ CACHE HIT ({similarity:.3f})</span>",
                        unsafe_allow_html=True
                    )

                st.markdown(response)
                chat["chat_history"].append({"role": "assistant", "content": response})

                if not cache_hit:
                    with st.expander("🔍 View Internal Process (For Portfolio)"):
                        tab1, tab2, tab3 = st.tabs(["Knowledge Graph", "HyDE Output", "Final Context Used"])
                        
                        tab1.markdown("**Successfully Extracted Connections:**")
                        tab1.code(graph_context if graph_context else "No clear relationships extracted based on your exact query.")
                        
                        tab2.markdown("**Hypothetical Document Generated:**")
                        tab2.write(hypothetical_doc)
                        
                        tab3.markdown("**Full Data sent to LLM:**")
                        tab3.write(final_context)

            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
