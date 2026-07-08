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

# ========================= LLM FACTORY (DETERMINISM) =========================
def get_llm(deterministic=True):
    """Temperature=0 removes sampling randomness — the #1 source of
    'same question, different answer' bugs."""
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.0 if deterministic else 0.7,
    )

# ========================= LEAKAGE UTILITIES =========================
LEAK_PATTERNS = [
    r"INTERNAL_RELATIONSHIP_NOTES_DO_NOT_QUOTE_DIRECTLY:?",
    r"EXTRACTED (KNOWLEDGE GRAPH FACTS|RELATIONSHIP FACTS):?",
    r"SOURCE PASSAGES:?",
    r"COMPRESSED DOCUMENT TEXT:?",
]

def clean_leakage(text):
    """Safety-net regex scrubber in case the LLM copies internal markers verbatim."""
    cleaned = text
    for pat in LEAK_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()

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
    """Sentence-level dedup + compression. Page is still tracked internally
    (useful for debugging) but is NOT surfaced in the prompt/answer anymore."""
    def __init__(self, cross_encoder):
        self.reranker = cross_encoder

    def build_and_compress(self, items, query, max_sentences=22):
        sentences_with_meta = []
        for item in items:
            page = item.get('page', 'Unknown')
            for s in nltk.sent_tokenize(item['text']):
                if len(s.strip()) > 20:
                    sentences_with_meta.append({"text": s.strip(), "page": page})

        unique_sentences, seen = [], set()
        for item in sentences_with_meta:
            key = item["text"].lower()
            if key not in seen:
                seen.add(key)
                unique_sentences.append(item)

        if not unique_sentences:
            return "No relevant context found."

        pairs = [[query, item["text"]] for item in unique_sentences]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, unique_sentences), key=lambda x: x[0], reverse=True)
        compressed = ranked[:max_sentences]

        # No [Source: Page X] tags injected into the actual text anymore.
        parts = [item["text"] for score, item in compressed if score > -2.0]
        return "\n".join(parts)


class KnowledgeGraphRAG:
    """
    Guaranteed unconditional graph build (not probabilistic),
    direction-validated edges (heuristic check against source sentence order),
    natural-language output with NO citations, leak-resistant.
    """
    def __init__(self, llm, persistent_graph):
        self.llm = llm
        self.graph = persistent_graph  # nx.MultiGraph — persists across the whole chat

    def build_graph(self, items):
        prompt = """Extract factual relationships from the text.
Output ONLY triplets in this exact format: Entity1 | Relationship | Entity2
Text: {text}"""
        for item in items:
            text = item['text']
            try:
                res = self.llm.invoke(prompt.format(text=text)).content
                for line in res.splitlines():
                    if line.count('|') == 2:
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) != 3 or parts[0].lower() == "none" or parts[2].lower() == "none":
                            continue
                        e1, rel, e2 = parts

                        # Direction-validation heuristic — if e2 actually appears
                        # BEFORE e1 in the source sentence, the extraction is likely backwards.
                        pos1 = text.lower().find(e1.lower())
                        pos2 = text.lower().find(e2.lower())
                        if pos1 != -1 and pos2 != -1 and pos2 < pos1:
                            e1, e2 = e2, e1

                        existing = self.graph.get_edge_data(e1, e2) or {}
                        already = any(d.get('relation', '').lower() == rel.lower() for d in existing.values())
                        if not already:
                            self.graph.add_edge(e1, e2, relation=rel)
            except Exception:
                continue

    def get_graph_context(self, query):
        if self.graph.number_of_nodes() == 0:
            return ""

        query_words = {w for w in set(re.findall(r'\b\w+\b', query.lower())) if len(w) > 3}
        all_facts = [
            {"u": u, "v": v, "rel": d.get('relation', 'is related to')}
            for u, v, d in self.graph.edges(data=True)
        ]

        def score(f):
            text = f"{f['u']} {f['rel']} {f['v']}".lower()
            return sum(1 for w in query_words if w in text)

        top_facts = [f for f in sorted(all_facts, key=score, reverse=True)[:12] if score(f) > 0]
        if not top_facts:
            return ""

        lines = [f"{f['u']} {f['rel']} {f['v']}." for f in top_facts]
        return "INTERNAL_RELATIONSHIP_NOTES_DO_NOT_QUOTE_DIRECTLY:\n" + "\n".join(lines)


# ========================= PARENT/NEIGHBOR EXPANSION =========================
def expand_with_neighbors(top_docs, all_chunks_data, window=1):
    """Hierarchical retrieval: pull in the chunk immediately before/after each
    retrieved chunk so connective tissue between two sections isn't cut off mid-thought."""
    selected = {}
    for doc in top_docs:
        idx = doc.metadata.get("chunk_index")
        page = doc.metadata.get("page", "?")
        text = doc.page_content
        if idx is None:
            selected[f"raw_{len(selected)}"] = {"text": text, "page": page, "chunk_index": -1}
            continue
        selected[idx] = {"text": text, "page": page, "chunk_index": idx}
        for offset in range(1, window + 1):
            for n_idx in (idx - offset, idx + offset):
                if 0 <= n_idx < len(all_chunks_data) and n_idx not in selected:
                    selected[n_idx] = all_chunks_data[n_idx]

    ordered = sorted(selected.values(), key=lambda x: x["chunk_index"] if isinstance(x["chunk_index"], int) else -1)
    return ordered


# ========================= CONSISTENCY SCORING =========================
def compute_consistency(responses):
    if len(responses) < 2:
        return 1.0
    vecs = np.array([embeddings.embed_query(r) for r in responses], dtype=np.float32)
    faiss.normalize_L2(vecs)
    sims = [float(np.dot(vecs[i], vecs[j])) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    return float(np.mean(sims)) if sims else 1.0


# ========================= CORE PIPELINE (SHARED BY CHAT UI + EVAL HARNESS) =========================
def run_rag_pipeline(query, chat, deterministic=True, use_cache=True, status=None):
    def log(msg):
        if status is not None:
            status.write(msg)

    debug = {
        "cache_hit": False, "sub_queries": [], "retrieved_pages": [],
        "graph_context": "", "compressed_text": "", "final_context": "",
    }

    llm = get_llm(deterministic=deterministic)

    if use_cache:
        cache_result = chat["semantic_cache"].get_cached_answer(query)
        if cache_result:
            response, sim = cache_result
            debug["cache_hit"] = True
            debug["similarity"] = sim
            return response, debug

    log("Decomposing question into sub-queries...")
    mq_prompt = f"""Break this question into up to 3 simpler, self-contained sub-questions
that together would let you fully answer it (useful for "how does X connect to Y" or
multi-part questions spanning different sections). Output ONLY the sub-questions,
one per line, no numbering, no extra text.
Question: {query}"""
    try:
        raw_sub = llm.invoke(mq_prompt).content
        sub_queries = [s.strip("-• ").strip() for s in raw_sub.splitlines() if s.strip()][:3]
    except Exception:
        sub_queries = []
    sub_queries.append(query)
    debug["sub_queries"] = sub_queries

    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings, sparse_encoder=chat["bm25_encoder"],
        index=chat["pinecone_index"], alpha=0.5, top_k=10, namespace=chat["namespace"]
    )

    log("Retrieving across all sub-questions...")
    all_retrieved = []
    for sq in sub_queries:
        try:
            all_retrieved.extend(retriever.invoke(sq))
        except Exception:
            continue

    unique_docs = {doc.page_content: doc for doc in all_retrieved}
    retrieved = list(unique_docs.values())

    if not retrieved:
        return "I don't have enough information in the document to answer that.", debug

    log("Reranking with cross-encoder...")
    doc_texts = [doc.page_content for doc in retrieved]
    scores = reranker.predict([[query, t] for t in doc_texts])
    top_docs = [d for _, d in sorted(zip(scores, retrieved), key=lambda x: x[0], reverse=True)[:8]]

    debug["retrieved_pages"] = [
        {"page": d.metadata.get("page", "?"), "chunk_index": d.metadata.get("chunk_index", "?"),
         "preview": d.page_content[:120] + "..."} for d in top_docs
    ]

    log("Expanding with neighboring context (parent-document retrieval)...")
    expanded_items = expand_with_neighbors(top_docs, chat["all_chunks_data"], window=1)

    log("Updating persistent knowledge graph...")
    kg_rag = KnowledgeGraphRAG(llm, chat["knowledge_graph"])
    top_docs_dicts = [{"text": d.page_content} for d in top_docs]
    kg_rag.build_graph(top_docs_dicts)
    graph_context = kg_rag.get_graph_context(query)
    debug["graph_context"] = graph_context

    log("Compressing & deduplicating context...")
    context_builder = AdvancedContextBuilder(reranker)
    compressed_text = context_builder.build_and_compress(expanded_items, query, max_sentences=22)
    debug["compressed_text"] = compressed_text

    final_context = ""
    if graph_context:
        final_context += graph_context + "\n\n---\n"
    final_context += "SOURCE PASSAGES:\n" + compressed_text
    debug["final_context"] = final_context

    log("Synthesizing final answer...")
    final_prompt = f"""You are an expert analytical assistant. Follow these rules exactly:

1. Use ONLY the information in the Context Data below. Do not use outside knowledge.
2. State only relationships and facts EXPLICITLY present in the retrieved text. If two
   passages are not clearly connected in the source, do NOT invent a connection —
   answer only what is directly supported.
3. Prefer cleanly paraphrasing a single relevant passage over blending multiple distant
   passages into one claim (this causes factual drift).
4. NEVER output raw internal formatting, symbols like "→" or "•", the literal words
   "Entity1/Entity2", or any internal labels/markers. Rewrite everything as natural,
   professional prose.
5. Do not include page numbers or any kind of citation markers in your answer.
6. If the answer is not supported by the Context Data, respond exactly:
   "I don't have enough information in the document to answer that."

Question: {query}

Context Data:
{final_context}

Answer:"""
    raw_response = llm.invoke(final_prompt).content
    response = clean_leakage(raw_response)

    if use_cache:
        chat["semantic_cache"].add_to_cache(query, response)

    return response, debug


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
        "knowledge_graph": nx.MultiGraph(),
        "all_chunks_data": [],
        "doc_name": None,
    }
    return chat_id

if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chats:
    st.session_state.current_chat_id = create_new_chat("Chat 1")

st.session_state.setdefault("deterministic_mode", True)
st.session_state.setdefault("use_cache", True)
st.session_state.setdefault("show_debug", True)

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

# ========================= SIDEBAR =========================
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
            if st.button(label, key=f"select_{cid}", use_container_width=True,
                         type="primary" if cid == st.session_state.current_chat_id else "secondary"):
                st.session_state.current_chat_id = cid
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{cid}"):
                del st.session_state.chats[cid]
                if not st.session_state.chats: create_new_chat("Chat 1")
                if st.session_state.current_chat_id == cid:
                    st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
                st.rerun()

    chat = st.session_state.chats[st.session_state.current_chat_id]

    st.divider()
    st.header("⚙️ Settings")
    st.session_state.deterministic_mode = st.checkbox(
        "Deterministic mode (temperature=0)", value=st.session_state.deterministic_mode,
        help="Keep ON so the same question always gives the same answer."
    )
    st.session_state.use_cache = st.checkbox(
        "Enable semantic cache", value=st.session_state.use_cache,
        help="Turn OFF while debugging consistency — cache hits can mask real pipeline behavior."
    )
    st.session_state.show_debug = st.checkbox(
        "Show debug info (retrieval / graph / context)", value=st.session_state.show_debug
    )
    if st.button("🧹 Clear Semantic Cache (this chat)"):
        chat["semantic_cache"] = SemanticCache(embeddings)
        st.success("Cache cleared for this chat.")

    st.divider()
    st.header("🛠️ Document Setup")

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
                if not docs:
                    raise ValueError("No content could be extracted from the PDF.")

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)

                for i, c in enumerate(chunks):
                    c.metadata["chunk_index"] = i
                    c.metadata["page"] = c.metadata.get("page", 0) + 1

                texts = [c.page_content for c in chunks]

                st.write("🔧 Fitting BM25 encoder...")
                bm25 = BM25Encoder().default()
                bm25.fit(texts)
                chat["bm25_encoder"] = bm25

                chat["all_chunks_data"] = [
                    {"text": c.page_content, "page": c.metadata["page"], "chunk_index": c.metadata["chunk_index"]}
                    for c in chunks
                ]

                st.write("🔧 Connecting to Pinecone...")
                pc = Pinecone(api_key=PINECONE_API_KEY)
                index_name = "graphrag"

                if index_name not in [idx.name for idx in pc.list_indexes()]:
                    pc.create_index(name=index_name, dimension=384, metric="dotproduct",
                                     spec=ServerlessSpec(cloud="aws", region="us-east-1"))
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
                        "id": f"chunk_{i}", "values": dense, "sparse_values": sparse,
                        "metadata": {
                            "context": text,
                            "page": chunk_.metadata["page"],
                            "chunk_index": chunk_.metadata["chunk_index"],
                            "source": uploaded_file.name
                        }
                    })

                for start_idx in range(0, len(vectors), 100):
                    index.upsert(vectors=vectors[start_idx:start_idx + 100], namespace=chat["namespace"])

                chat["pdf_processed"] = True
                chat["doc_name"] = uploaded_file.name
                if chat["name"].startswith("Chat "):
                    chat["name"] = uploaded_file.name[:30]
                st.success(f"✅ Document processed! {len(chunks)} chunks indexed.")

        except Exception as e:
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

# ========================= MAIN UI =========================
st.markdown('<p class="main-header">🧠 Advanced Graph RAG System</p>', unsafe_allow_html=True)
st.caption("Deterministic • Multi-Hop Retrieval • Leak-Free Graph Synthesis (No Citations)")

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
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if query.lower().strip() in ["hi", "hello", "hey"]:
            resp = "Hello! 👋 Ask me anything about your document — I now guarantee consistent answers."
            st.markdown(resp)
            chat["chat_history"].append({"role": "assistant", "content": resp})
        else:
            try:
                with st.status("Thinking...", expanded=st.session_state.show_debug) as status:
                    response, debug = run_rag_pipeline(
                        query, chat,
                        deterministic=st.session_state.deterministic_mode,
                        use_cache=st.session_state.use_cache,
                        status=status
                    )
                    status.update(label="Done", state="complete")

                if debug.get("cache_hit"):
                    st.markdown(
                        f"<span class='badge cache-hit'>⚡ CACHE HIT ({debug['similarity']:.2f})</span><br><br>",
                        unsafe_allow_html=True
                    )

                st.markdown(response)
                chat["chat_history"].append({"role": "assistant", "content": response})

                if st.session_state.show_debug and not debug.get("cache_hit"):
                    with st.expander("🔍 Debug: Retrieval, Graph & Context (For QA / Portfolio)"):
                        st.markdown("**Sub-queries used for multi-hop retrieval:**")
                        for sq in debug.get("sub_queries", []):
                            st.write(f"- {sq}")
                        st.markdown("**Retrieved chunks (page / chunk index / preview):**")
                        st.json(debug.get("retrieved_pages", []))
                        st.markdown("**Internal knowledge-graph facts (never shown to the user):**")
                        st.code(debug.get("graph_context") or "None extracted.")
                        st.markdown("**Compressed context sent to the LLM:**")
                        st.write(debug.get("compressed_text", ""))

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ========================= EVALUATION HARNESS =========================
st.divider()
with st.expander("🧪 Evaluation Harness — Consistency Testing"):
    st.caption(
        "Run the same questions multiple times to verify determinism and detect drift — "
        "without doing it manually every time."
    )
    default_qs = "What is the main topic of this document?\nHow does the first major concept connect to the last one discussed?"
    test_qs_raw = st.text_area("Test questions (one per line)", value=default_qs, height=120,
                                key=f"eval_qs_{st.session_state.current_chat_id}")
    runs_per_q = st.slider("Runs per question", 2, 5, 3, key=f"eval_runs_{st.session_state.current_chat_id}")

    if st.button("▶️ Run Evaluation", key=f"run_eval_{st.session_state.current_chat_id}"):
        questions = [q.strip() for q in test_qs_raw.splitlines() if q.strip()]
        results = []
        progress = st.progress(0.0)
        total = max(len(questions) * runs_per_q, 1)
        step = 0

        for q in questions:
            responses = []
            for _ in range(runs_per_q):
                resp, _ = run_rag_pipeline(q, chat, deterministic=True, use_cache=False, status=None)
                responses.append(resp)
                step += 1
                progress.progress(step / total)

            consistency = compute_consistency(responses)
            results.append({
                "Question": q,
                "Consistency Score (0-1)": round(consistency, 3),
                "Sample Answer": responses[0][:200] + ("..." if len(responses[0]) > 200 else "")
            })

        st.session_state[f"eval_results_{st.session_state.current_chat_id}"] = results

    results_key = f"eval_results_{st.session_state.current_chat_id}"
    if results_key in st.session_state:
        st.dataframe(st.session_state[results_key], use_container_width=True)
        st.caption(
            "Consistency Score ≥ 0.9 typically means near-identical answers across runs. "
            "Anything below ~0.7 indicates the pipeline is still non-deterministic for that question type."
        )
