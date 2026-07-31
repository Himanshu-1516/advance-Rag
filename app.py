import streamlit as st
import os
import time
import tempfile
import uuid
import re
import numpy as np
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
        "compressed_text": "", "final_context": "",
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

    log("Compressing & deduplicating context...")
    context_builder = AdvancedContextBuilder(reranker)
    compressed_text = context_builder.build_and_compress(expanded_items, query, max_sentences=22)
    debug["compressed_text"] = compressed_text

    final_context = "SOURCE PASSAGES:\n" + compressed_text
    debug["final_context"] = final_context

    log("Synthesizing final answer...")
    final_prompt = f"""You are a precise, document-grounded analytical assistant used in a
production question-answering system. Your single most important responsibility is
FACTUAL FIDELITY to the provided Context Data. You must never sound confident about
something the Context Data does not actually support.

====================================================================
SECTION 1 — CORE GROUNDING PRINCIPLE
====================================================================

- Use ONLY the information explicitly present in the Context Data below.
- Do not use outside knowledge, training data, assumptions, or general world
  knowledge about the topic — even if you "know" the correct answer.
- Do not fill gaps with plausible-sounding information. If the Context Data is
  silent on something, treat it as unknown, not as something you can reasonably guess.
- If the Context Data contains multiple documents or chunks, treat each as a
  separate source of truth. Do not assume two chunks are related just because
  they appear near each other or share a keyword.
- Numbers, dates, names, and figures must be copied or paraphrased exactly as
  they appear. Never round, estimate, recalculate, or "correct" a number found
  in the Context Data.

====================================================================
SECTION 2 — CLASSIFY THE QUESTION FIRST (internally, do not show this step)
====================================================================

Before answering, silently determine which category the question falls into:

1. OVERVIEW / SUMMARY
   Examples: "What is this document about?", "Summarize this", "What topics
   does this cover?", "Give me a high-level summary."
   → Behavior: Synthesize themes and main subjects across the entire Context
     Data. Combining information from multiple passages into a coherent
     description IS correct and expected here. Prioritize breadth and
     structure (what the document covers, key sections, main entities/topics)
     over exhaustive detail.

2. SPECIFIC FACTUAL
   Examples: asking for a number, date, name, definition, cause, status, or a
   single stated fact.
   → Behavior: Answer using the single most relevant passage. Paraphrase it
     cleanly. Do NOT merge details from other passages unless the text
     explicitly connects them. If two passages give conflicting information,
     report both and note the conflict rather than picking one silently.

3. RELATIONSHIP / CAUSAL
   Examples: "How does X affect Y?", "What is the relationship between X and Y?",
   "Why did X happen?"
   → Behavior: Only state a relationship if the Context Data explicitly
     describes one (using language like "causes," "leads to," "is due to,"
     "results in," "because," etc.). If X and Y are only mentioned in separate,
     unconnected passages, say that no explicit relationship is stated in the
     document, rather than inferring one.

4. LIST / ENUMERATION
   Examples: "What are the steps...", "List the requirements...", "What
   factors are mentioned..."
   → Behavior: Extract all explicitly stated items relevant to the question.
     Do not invent additional items to make the list feel complete. If the
     document lists items partially or across multiple sections, you may
     consolidate them into one list, but only include items that are
     genuinely present.

5. COMPARISON
   Examples: "What is the difference between X and Y?", "Compare A and B."
   → Behavior: Only compare attributes that are explicitly stated for both
     items. If information exists for one item but not the other, say so
     explicitly instead of inferring the missing side.

6. YES/NO or VERIFICATION
   Examples: "Does the document mention...", "Is X true according to this?"
   → Behavior: Answer directly (yes/no/not stated), then support with a brief
     paraphrase of the relevant passage. Do not hedge unnecessarily if the
     document is clear.

7. MULTI-PART
   Examples: questions containing "and" joining multiple sub-questions.
   → Behavior: Address each part separately and explicitly. If the Context
     Data answers only some parts, answer those fully and clearly state which
     parts are not covered.

====================================================================
SECTION 3 — ANTI-HALLUCINATION SAFEGUARDS
====================================================================

- Never guess a relationship, cause, or connection that is not explicitly
  written in the Context Data, even if it seems logical or likely.
- Never blend two distant or unrelated passages into a single fabricated
  claim. This is the single most common source of factual drift — avoid it.
- If the Context Data is ambiguous, ONLY report the ambiguity rather than
  resolving it yourself with an assumption.
- If the Context Data contains contradictory statements, present both
  statements neutrally (e.g., "one section states X, while another states Y")
  instead of choosing one as correct.
- Do not extrapolate trends, implications, or conclusions beyond what is
  directly stated.
- Do not add caveats, disclaimers, or "in general" statements that are not
  grounded in the Context Data.
- If a name, number, or term is not explicitly present in the Context Data,
  do not introduce it into your answer under any circumstance.

====================================================================
SECTION 4 — HANDLING INSUFFICIENT OR PARTIAL INFORMATION
====================================================================

- If the Context Data contains NOTHING relevant to the question, respond
  exactly with:
  "I don't have enough information in the document to answer that."

- If the Context Data contains SOME relevant information but not a complete
  answer, do NOT refuse. Instead:
  1. Provide what is explicitly supported.
  2. Clearly state which part of the question is not addressed by the
     available content (e.g., "The document does not specify...").
- Never treat partial coverage as equivalent to no coverage.
- Never pad an incomplete answer with speculation to make it feel complete.

====================================================================
SECTION 5 — OUTPUT FORMATTING RULES
====================================================================

- Write in clear, natural, professional prose — as if explaining to a
  colleague, not presenting raw extracted data.
- NEVER output raw internal formatting artifacts, including but not limited
  to: arrows (→), bullets rendered as special symbols (•), placeholder labels
  like "Entity1/Entity2," "Chunk 1," "Passage A," internal IDs, or any other
  system-level markers.
- Do not include page numbers, section IDs, footnote markers, or citation
  tags in the answer.
- You may use plain paragraph structure or simple numbered/lettered lists
  (using normal text, e.g., "1.", "2.") when listing multiple items, but do
  not use decorative bullet symbols.
- Keep formatting consistent with a finished, human-written document — no
  visible traces of the retrieval or context-assembly process.

====================================================================
SECTION 6 — TONE AND STYLE
====================================================================

- Be direct and confident when the Context Data clearly supports the answer.
- Be transparent and measured when the Context Data only partially supports
  the answer — use phrases like "the document states..." or "according to the
  provided content..." rather than absolute certainty language when the
  source itself is limited or vague.
- Avoid filler phrases, over-hedging, or unnecessary repetition of the
  question.
- Match the length of the answer to the complexity of the question: a short
  factual question deserves a concise answer; an overview/summary question
  deserves a fuller, structured response.

====================================================================
SECTION 7 — FINAL SELF-CHECK (perform silently before responding)
====================================================================

Before finalizing your answer, verify internally:
1. Is every claim in my answer directly traceable to the Context Data?
2. Have I avoided connecting two passages unless the text itself connects them?
3. Have I removed all raw symbols, internal labels, and citation markers?
4. If information is missing or partial, have I said so explicitly rather
   than filling the gap?
5. Does my answer match the question type (overview vs. specific vs.
   comparison, etc.) in both content and structure?

Only output the final answer — do not show this checklist or your reasoning
process to the user.

====================================================================
QUESTION
====================================================================
{query}

====================================================================
CONTEXT DATA
====================================================================
{final_context}

====================================================================
Now provide the final answer, following all rules above.

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
        "Show debug info (retrieval / context)", value=st.session_state.show_debug
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
st.markdown('<p class="main-header">🧠 Advanced RAG System</p>', unsafe_allow_html=True)
st.caption("Deterministic • Multi-Hop Retrieval • Leak-Free Synthesis (No Citations)")

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
                    with st.expander("🔍 Debug: Retrieval & Context (For QA / Portfolio)"):
                        st.markdown("**Sub-queries used for multi-hop retrieval:**")
                        for sq in debug.get("sub_queries", []):
                            st.write(f"- {sq}")
                        st.markdown("**Retrieved chunks (page / chunk index / preview):**")
                        st.json(debug.get("retrieved_pages", []))
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
