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
    """Handles Context Compression, Token Reduction, and Deduplication with Citation Anchoring."""
    def __init__(self, cross_encoder):
        self.reranker = cross_encoder

    def build_and_compress(self, top_docs, query, max_sentences=20):
        sentences_with_meta = []
        for doc in top_docs:
            page = doc.metadata.get('page', 'Unknown')
            sents = nltk.sent_tokenize(doc.page_content)
            for s in sents:
                if len(s.strip()) > 20:
                    sentences_with_meta.append({"text": s.strip(), "page": page})

        unique_sentences = []
        seen = set()
        for item in sentences_with_meta:
            clean_text = item["text"].lower()
            if clean_text not in seen:
                seen.add(clean_text)
                unique_sentences.append(item)

        if not unique_sentences:
            return "No relevant context found."

        pairs = [[query, item["text"]] for item in unique_sentences]
        scores = self.reranker.predict(pairs)

        scored_sentences = zip(scores, unique_sentences)
        ranked_sentences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)
        compressed_data = ranked_sentences[:max_sentences]

        final_context_parts = []
        for score, item in compressed_data:
            if score > -2.0:
                final_context_parts.append(f"[Source: Page {item['page']}] {item['text']}")

        return "\n".join(final_context_parts)


class KnowledgeGraphRAG:
    """
    FIXED: Uses a persistent MultiGraph (supports multiple relations per entity pair
    across different chapters/pages) and outputs NATURAL LANGUAGE facts instead of
    raw arrow-symbol fragments to prevent 'graph-fact leakage' into the final answer.
    """
    def __init__(self, llm, persistent_graph):
        self.llm = llm
        self.graph = persistent_graph  # nx.MultiGraph - persists across the whole chat session

    def build_graph(self, documents):
        prompt = """Extract factual relationships from the text.
Output ONLY triplets in this exact format: Entity1 | Relationship | Entity2
Text: {text}"""
        for doc in documents:
            page = doc.metadata.get('page', '?')
            try:
                res = self.llm.invoke(prompt.format(text=doc.page_content)).content
                for line in res.splitlines():
                    if line.count('|') == 2:
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) == 3 and parts[0].lower() != "none" and parts[2].lower() != "none":
                            # Avoid adding exact duplicate relation for same pair
                            existing = self.graph.get_edge_data(parts[0], parts[2]) or {}
                            already_exists = any(
                                d.get('relation', '').lower() == parts[1].lower()
                                for d in existing.values()
                            )
                            if not already_exists:
                                self.graph.add_edge(parts[0], parts[2], relation=parts[1], page=page)
            except Exception:
                continue

    def get_graph_context(self, query):
        if self.graph.number_of_nodes() == 0:
            return ""

        query_words = {w for w in set(re.findall(r'\b\w+\b', query.lower())) if len(w) > 3}

        all_facts = []
        for u, v, data in self.graph.edges(data=True):
            rel = data.get('relation', 'is related to')
            page = data.get('page', '?')
            all_facts.append({"u": u, "v": v, "rel": rel, "page": page})

        def fact_score(fact):
            text = f"{fact['u']} {fact['rel']} {fact['v']}".lower()
            return sum(1 for w in query_words if w in text)

        scored_facts = sorted(all_facts, key=fact_score, reverse=True)
        top_facts = [f for f in scored_facts[:15] if fact_score(f) > 0]

        if not top_facts:
            return ""

        # ✅ FIX: Natural language sentence format (NOT raw arrow/bullet symbols)
        # This prevents the LLM from copy-pasting a clunky "•  X → (Y) → Z" fragment
        # directly into its answer. It now reads like a real, citable sentence.
        sentences = []
        for f in top_facts:
            sentences.append(f"{f['u']} {f['rel']} {f['v']} (Page {f['page']}).")

        return "RELATIONSHIP FACTS (write these naturally into full sentences, do not copy raw format):\n" + "\n".join(sentences)


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
        "knowledge_graph": nx.MultiGraph(),  # ✅ Upgraded from Graph -> MultiGraph
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
st.caption("Multi-Query Synthesis + Persistent Graph + Strict Citations (Leak-Free)")

chat = st.session_state.chats[st.session_state.current_chat_id]
st.subheader(f"💬 {chat['name']}" + (f"  ·  📄 {chat['doc_name']}" if chat["doc_name"] else ""))

if not chat["pdf_processed"]:
    st.info("👈 Upload a PDF for this chat in the sidebar, then click **Process Document** to start chatting.")
    st.stop()

for message in chat["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask a complex, multi-part question...")

if query:
    chat["chat_history"].append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)

    with st.chat_message("assistant"):
        if query.lower().strip() in ["hi", "hello", "hey"]:
            resp = "Hello! 👋 I am equipped for cross-chapter synthesis and relationship mapping. What would you like to know?"
            st.markdown(resp)
            chat["chat_history"].append({"role": "assistant", "content": resp})
        else:
            try:
                llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)
                retriever = PineconeHybridSearchRetriever(
                    embeddings=embeddings, sparse_encoder=chat["bm25_encoder"],
                    index=chat["pinecone_index"], alpha=0.5, top_k=6, namespace=chat["namespace"]
                )
                kg_rag = KnowledgeGraphRAG(llm, chat["knowledge_graph"])
                context_builder = AdvancedContextBuilder(reranker)

                with st.status("Thinking...", expanded=True) as status:
                    cache_result = chat["semantic_cache"].get_cached_answer(query)

                    if cache_result:
                        response, sim = cache_result
                        status.update(label=f"Answered from Cache (sim: {sim:.2f})", state="complete")
                    else:
                        status.write("Decomposing question for cross-chapter synthesis...")
                        mq_prompt = f"Break this complex question down into 3 simpler sub-queries to ensure we find all necessary context across a whole document.\nQuestion: {query}\nOutput ONLY the 3 sub-queries, one per line."
                        sub_queries = llm.invoke(mq_prompt).content.splitlines()
                        sub_queries = [q.strip() for q in sub_queries if q.strip()][:3]
                        sub_queries.append(query)

                        status.write("Retrieving across multiple logical pathways...")
                        all_retrieved = []
                        for sq in sub_queries:
                            all_retrieved.extend(retriever.invoke(sq))

                        unique_docs = {doc.page_content: doc for doc in all_retrieved}
                        retrieved = list(unique_docs.values())

                        if retrieved:
                            status.write("Reranking multi-source documents...")
                            doc_texts = [doc.page_content for doc in retrieved]
                            scores = reranker.predict([[query, text] for text in doc_texts])
                            top_docs = [doc for _, doc in sorted(zip(scores, retrieved), key=lambda x: x[0], reverse=True)[:8]]

                            status.write("Updating Persistent Graph & Extracting Relationships...")
                            kg_rag.build_graph(top_docs)
                            graph_context = kg_rag.get_graph_context(query)

                            status.write("Compressing context for citation anchoring...")
                            compressed_text = context_builder.build_and_compress(top_docs, query, max_sentences=20)

                            final_context = ""
                            if graph_context:
                                final_context += graph_context + "\n\n---\n"
                            final_context += "COMPRESSED DOCUMENT TEXT:\n" + compressed_text
                        else:
                            final_context = "No relevant context found."

                        status.update(label="Synthesizing Final Answer...", state="running")

                        # ✅ FIX: Explicit anti-leakage guardrail added to the prompt
                        final_prompt = f"""You are an expert analytical assistant. Follow these rules strictly:

1. SYNTHESIZE: If the question asks about relationships or requires cross-chapter logic, weave the concepts from the Context and Relationship Facts into clear, natural, professional prose.
2. FORMATTING RULE (VERY IMPORTANT): NEVER use raw symbols like "→", bullet points ("•"), or the literal words "Entity1/Entity2" in your answer. Do not copy the raw fact list format. Rewrite every relationship as a complete, grammatically correct sentence.
3. CITATION MANDATE: Every factual claim must end with an inline citation formatted exactly like this: (Page X). Use ONLY the page numbers provided in the [Source: Page X] tags or the "(Page X)" tags in the Relationship Facts. Never invent a page number.
4. If the answer isn't in the context, say: "I don't have enough information in the document to answer that."

Question: {query}

Context Data:
{final_context}

Write a polished, professional, citation-backed answer below:"""

                        response = llm.invoke(final_prompt).content
                        chat["semantic_cache"].add_to_cache(query, response)
                        status.update(label="Done", state="complete")

                if cache_result:
                    st.markdown(f"<span class='badge cache-hit'>⚡ CACHE HIT ({sim:.2f})</span><br><br>", unsafe_allow_html=True)

                st.markdown(response)
                chat["chat_history"].append({"role": "assistant", "content": response})

                if not cache_result:
                    with st.expander("🔍 View Synthesis Data (For Portfolio)"):
                        st.markdown("**1. Sub-Queries Generated (Multi-Hop):**")
                        for sq in sub_queries:
                            st.write(f"- {sq}")
                        st.markdown("**2. Persistent Graph Relationships Extracted (Raw, Internal Only):**")
                        st.code(graph_context if graph_context else "None extracted.")
                        st.markdown("**3. Strict Citation-Anchored Context:**")
                        st.write(compressed_text)

            except Exception as e:
                st.error(f"Error: {str(e)}")
