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

# Download NLTK data required for BM25 and Sentence Tokenization
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
</style>
""", unsafe_allow_html=True)

if not KEYS_CONFIGURED:
    st.error("⚠️ API keys are not configured yet. Please set `GROQ_API_KEY` and `PINECONE_API_KEY` in the script.")
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

    def get_cached_answer(self, query):
        if self.index.ntotal == 0:
            return None
        vec = np.array([self.embeddings.embed_query(query)], dtype=np.float32)
        faiss.normalize_L2(vec)
        distances, indices = self.index.search(vec, 1)
        if distances[0][0] >= self.threshold:
            return self.cache_answers[indices[0][0]], float(distances[0][0])
        return None

    def add_to_cache(self, query, answer):
        vec = np.array([self.embeddings.embed_query(query)], dtype=np.float32)
        faiss.normalize_L2(vec)
        self.index.add(vec)
        self.cache_answers.append(answer)


class AdvancedContextBuilder:
    """Handles Context Compression, Token Reduction, and Deduplication"""
    def __init__(self, cross_encoder):
        self.reranker = cross_encoder

    def build_and_compress(self, top_docs, query, max_sentences=12):
        # 1. Break into individual sentences
        sentences_with_meta = []
        for doc in top_docs:
            page = doc.metadata.get('page', '?')
            sents = nltk.sent_tokenize(doc.page_content)
            for s in sents:
                if len(s.strip()) > 20: # Ignore useless short fragments
                    sentences_with_meta.append({"text": s.strip(), "page": page})

        # 2. Deduplication (Exact & Substring match to fix chunk overlap)
        unique_sentences = []
        seen = set()
        for item in sentences_with_meta:
            clean_text = item["text"].lower()
            if clean_text not in seen:
                seen.add(clean_text)
                unique_sentences.append(item)

        if not unique_sentences:
            return "No relevant context found."

        # 3. Context Compression (Score specific sentences against query)
        pairs = [[query, item["text"]] for item in unique_sentences]
        scores = self.reranker.predict(pairs)
        
        scored_sentences = zip(scores, unique_sentences)
        
        # 4. Token Reduction: Sort by score and keep only the top N sentences
        ranked_sentences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)
        compressed_data = ranked_sentences[:max_sentences]
        
        # 5. Reassemble logically for the LLM
        page_groups = {}
        for score, item in compressed_data:
            if score < -2.0: # Filter out wildly irrelevant sentences
                continue
            p = item["page"]
            if p not in page_groups:
                page_groups[p] = []
            page_groups[p].append(item["text"])
            
        final_context_parts = []
        for p, sents in page_groups.items():
            final_context_parts.append(f"[Page {p}]: " + " ".join(sents))
            
        return "\n\n".join(final_context_parts)


class KnowledgeGraphRAG:
    def __init__(self, llm):
        self.llm = llm
        self.graph = nx.Graph()

    def build_graph(self, documents):
        prompt = """Extract factual relationships from the text.
Output ONLY triplets in this exact format: Entity1 | Relationship | Entity2
Text: {text}"""
        for doc in documents:
            try:
                res = self.llm.invoke(prompt.format(text=doc.page_content)).content
                for line in res.splitlines():
                    if line.count('|') == 2:
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) == 3 and parts[0].lower() != "none":
                            self.graph.add_edge(parts[0], parts[2], relation=parts[1])
            except Exception:
                continue

    def get_graph_context(self, query):
        if self.graph.number_of_nodes() == 0:
            return ""
        
        query_words = {w for w in set(re.findall(r'\b\w+\b', query.lower())) if len(w) > 3}
        all_edges = [(u, v, data.get('relation', '')) for u, v, data in self.graph.edges(data=True)]

        def edge_score(edge):
            text = f"{edge[0]} {edge[1]} {edge[2]}".lower()
            return sum(1 for w in query_words if w in text)

        scored_edges = sorted(all_edges, key=edge_score, reverse=True)
        relations = [f"• {u} → ({rel}) → {v}" for u, v, rel in scored_edges[:15] if edge_score((u, v, rel)) > 0]

        return "EXTRACTED KNOWLEDGE GRAPH FACTS:\n" + "\n".join(relations) if relations else ""


# ========================= MULTI-CHAT SESSION STATE =========================
def create_new_chat(name=None):
    chat_id = f"chat_{uuid.uuid4().hex[:8]}"
    st.session_state.chats[chat_id] = {
        "name": name or f"Chat {len(st.session_state.chats) + 1}",
        "chat_history": [],
        "pdf_processed": False,
        "bm25_encoder": None,
        "pinecone_index": None,
        "namespace": chat_id,
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
        status = pc.describe_index(index_name).status
        if status.ready: return True
        if time.time() - start > timeout: raise TimeoutError()
        time.sleep(1)

# ========================= SIDEBAR: CHAT MANAGEMENT =========================
with st.sidebar:
    st.header("💬 Chats")
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_chat_id = create_new_chat()
        st.rerun()

    st.divider()
    for cid, cdata in list(st.session_state.chats.items()):
        col1, col2 = st.columns([5, 1])
        with col1:
            label = ("📄 " if cdata["pdf_processed"] else "🗒️ ") + cdata["name"]
            if st.button(label, key=f"select_{cid}", use_container_width=True, type="primary" if cid == st.session_state.current_chat_id else "secondary"):
                st.session_state.current_chat_id = cid
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{cid}"):
                del st.session_state.chats[cid]
                if not st.session_state.chats: create_new_chat("Chat 1")
                if st.session_state.current_chat_id == cid: st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
                st.rerun()

    st.divider()
    st.header("🛠️ Document Setup")
    chat = st.session_state.chats[st.session_state.current_chat_id]

    new_name = st.text_input("Chat name", value=chat["name"], key=f"name_{st.session_state.current_chat_id}")
    if new_name and new_name != chat["name"]:
        chat["name"] = new_name

    uploaded_file = st.file_uploader("Upload PDF", type="pdf", key=f"upload_{st.session_state.current_chat_id}")

    if st.button("Process Document", type="primary", disabled=uploaded_file is None):
        tmp_path = None
        try:
            with st.spinner("Processing PDF + Building Hybrid Index..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)
                texts = [chunk.page_content for chunk in chunks]

                st.write("🔧 Fitting BM25 encoder...")
                bm25 = BM25Encoder().default()
                bm25.fit(texts)
                chat["bm25_encoder"] = bm25

                pc = Pinecone(api_key=PINECONE_API_KEY)
                index_name = "graphrag"

                if index_name not in [idx.name for idx in pc.list_indexes()]:
                    pc.create_index(name=index_name, dimension=384, metric="dotproduct", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
                    wait_for_index_ready(pc, index_name)
                
                index = pc.Index(index_name)
                chat["pinecone_index"] = index

                vectors = []
                for i, (text, chunk_) in enumerate(zip(texts, chunks)):
                    dense = embeddings.embed_query(text)
                    sparse = bm25.encode_documents([text])[0]
                    vectors.append({
                        "id": f"chunk_{i}", "values": dense, "sparse_values": sparse,
                        "metadata": {"context": text, "page": chunk_.metadata.get("page", 0)}
                    })

                for start_idx in range(0, len(vectors), 100):
                    index.upsert(vectors=vectors[start_idx:start_idx + 100], namespace=chat["namespace"])

                chat["pdf_processed"] = True
                chat["doc_name"] = uploaded_file.name
                if chat["name"].startswith("Chat "): chat["name"] = uploaded_file.name[:30]
                st.success("✅ Document processed!")
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            if tmp_path and os.path.exists(tmp_path): os.unlink(tmp_path)
        if chat["pdf_processed"]: st.rerun()

# ========================= MAIN UI =========================
st.markdown('<p class="main-header">🧠 Advanced Graph RAG System</p>', unsafe_allow_html=True)
st.caption("HyDE + Hybrid Reranking + Context Compression + Token Reduction + Knowledge Graph")

chat = st.session_state.chats[st.session_state.current_chat_id]
st.subheader(f"💬 {chat['name']}" + (f"  ·  📄 {chat['doc_name']}" if chat["doc_name"] else ""))

if not chat["pdf_processed"]:
    st.info("👈 Upload a PDF for this chat in the sidebar, then click **Process Document** to start chatting.")
    st.stop()

for message in chat["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask any question about your document...")

if query:
    chat["chat_history"].append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)

    with st.chat_message("assistant"):
        if query.lower().strip() in ["hi", "hello", "hey"]:
            resp = "Hello! 👋 I am your Advanced Graph RAG Assistant. What would you like to know?"
            st.markdown(resp)
            chat["chat_history"].append({"role": "assistant", "content": resp})
        else:
            try:
                llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)
                retriever = PineconeHybridSearchRetriever(
                    embeddings=embeddings, sparse_encoder=chat["bm25_encoder"],
                    index=chat["pinecone_index"], alpha=0.5, top_k=8, namespace=chat["namespace"]
                )
                kg_rag = KnowledgeGraphRAG(llm)
                context_builder = AdvancedContextBuilder(reranker) # INIT NEW BUILDER

                with st.status("Thinking...", expanded=True) as status:
                    cache_result = chat["semantic_cache"].get_cached_answer(query)
                    if cache_result:
                        response, sim = cache_result
                        status.update(label=f"Answered from Cache (sim: {sim:.2f})", state="complete")
                        st.markdown(response)
                        chat["chat_history"].append({"role": "assistant", "content": response})
                    else:
                        status.write("Generating HyDE...")
                        hyde_doc = llm.invoke(f"Answer factually: {query}").content

                        status.write("Retrieving & Reranking Documents...")
                        retrieved = retriever.invoke(hyde_doc)

                        if retrieved:
                            doc_texts = [doc.page_content for doc in retrieved]
                            scores = reranker.predict([[hyde_doc, text] for text in doc_texts])
                            top_docs = [doc for _, doc in sorted(zip(scores, retrieved), key=lambda x: x[0], reverse=True)[:5]]

                            status.write("Extracting Graph & Compressing Context...")
                            kg_rag.build_graph(top_docs)
                            graph_context = kg_rag.get_graph_context(query)
                            
                            # ✨ NEW CONTEXT COMPRESSION IN ACTION ✨
                            compressed_text = context_builder.build_and_compress(top_docs, query)

                            final_context = ""
                            if graph_context: final_context += graph_context + "\n\n---\n"
                            final_context += "COMPRESSED DOCUMENT TEXT:\n" + compressed_text
                        else:
                            final_context = "No relevant context found."

                        status.update(label="Generating Final Answer...", state="running")
                        prompt = f"Answer using ONLY this context. If not found, say so.\nQuestion: {query}\n\nContext:\n{final_context}"
                        
                        response = llm.invoke(prompt).content
                        chat["semantic_cache"].add_to_cache(query, response)
                        status.update(label="Done", state="complete")

                        st.markdown(response)
                        chat["chat_history"].append({"role": "assistant", "content": response})

                        with st.expander("🔍 View Context Compression (For Portfolio)"):
                            st.markdown("**1. Tokens Reduced & Deduplicated Context:**")
                            st.write(compressed_text)
                            st.markdown("**2. Knowledge Graph Connections:**")
                            st.code(graph_context if graph_context else "None extracted.")

            except Exception as e:
                st.error(f"Error: {str(e)}")
