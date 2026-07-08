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

# ========================= API KEYS & DEFAULT FALLBACKS =========================
DEFAULT_GROQ_API_KEY = "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"
DEFAULT_PINECONE_API_KEY = "pcsk_39EGLB_PC9i9y7MQo2FxSqgqdX4akFP3LPFoNqHirwHsicYqAivgQASB4bFsM9ocPY9epZ"

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", DEFAULT_GROQ_API_KEY)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY", DEFAULT_PINECONE_API_KEY)
GROQ_MODEL = os.getenv("GROQ_MODEL") or st.secrets.get("GROQ_MODEL", "llama-3.1-8b-instant")

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
    def __init__(self, embeddings_model, threshold=0.92):
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
    """Handles Context Compression, Sentence-Level Reranking, and Deduplication."""
    def __init__(self, cross_encoder):
        self.reranker = cross_encoder

    def build_and_compress(self, top_docs, query, max_sentences=25):
        sentences = []
        for doc in top_docs:
            sents = nltk.sent_tokenize(doc.page_content)
            for s in sents:
                if len(s.strip()) > 15:
                    sentences.append(s.strip())

        unique_sentences = []
        seen = set()
        for s in sentences:
            clean_text = s.lower()
            if clean_text not in seen:
                seen.add(clean_text)
                unique_sentences.append(s)

        if not unique_sentences:
            return "No relevant context found."

        pairs = [[query, s] for s in unique_sentences]
        scores = self.reranker.predict(pairs)

        scored_sentences = zip(scores, unique_sentences)
        ranked_sentences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)
        compressed_data = ranked_sentences[:max_sentences]

        final_context_parts = []
        for score, text in compressed_data:
            if score > -3.0:
                final_context_parts.append(text)

        return "\n".join(final_context_parts)


class KnowledgeGraphRAG:
    """
    Constructs declarative semantic assertions to eliminate leakage of Graph syntax
    (raw symbols, arrow formats) into final user-facing responses.
    """
    def __init__(self, llm, persistent_graph):
        self.llm = llm
        self.graph = persistent_graph

    def build_graph(self, documents):
        prompt = """Extract clear declarative facts from this text. 
Output ONLY triplets in this strict format: Entity1 | Relationship | Entity2
Do not use numbered lists, bullet points, or special characters.
Text: {text}"""
        for doc in documents:
            try:
                res = self.llm.invoke(prompt.format(text=doc.page_content)).content
                for line in res.splitlines():
                    if line.count('|') == 2:
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) == 3 and all(parts):
                            existing = self.graph.get_edge_data(parts[0], parts[2]) or {}
                            already_exists = any(
                                d.get('relation', '').lower() == parts[1].lower()
                                for d in existing.values()
                            )
                            if not already_exists:
                                self.graph.add_edge(parts[0], parts[2], relation=parts[1])
            except Exception:
                continue

    def get_graph_context(self, query):
        if self.graph.number_of_nodes() == 0:
            return ""

        query_words = {w for w in set(re.findall(r'\b\w+\b', query.lower())) if len(w) > 3}

        all_facts = []
        for u, v, data in self.graph.edges(data=True):
            rel = data.get('relation', 'is connected to')
            all_facts.append({"u": u, "v": v, "rel": rel})

        def fact_score(fact):
            text = f"{fact['u']} {fact['rel']} {fact['v']}".lower()
            return sum(1 for w in query_words if w in text)

        scored_facts = sorted(all_facts, key=fact_score, reverse=True)
        top_facts = [f for f in scored_facts[:12] if fact_score(f) > 0]

        if not top_facts:
            return ""

        sentences = []
        for f in top_facts:
            sentences.append(f"Fact: {f['u']} {f['rel']} {f['v']}.")

        return "EXTRACTED SYSTEM RELATIONSHIPS (Convert these into natural statements, do not copy structural markers):\n" + "\n".join(sentences)


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
    st.header("💬 Conversations")
    if st.button("➕ Create New Chat", use_container_width=True):
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

    new_name = st.text_input("Chat display name", value=chat["name"], key=f"name_{st.session_state.current_chat_id}")
    if new_name and new_name != chat["name"]:
        chat["name"] = new_name

    uploaded_file = st.file_uploader("Upload Document Source", type="pdf", key=f"upload_{st.session_state.current_chat_id}")

    if st.button("Process Document Context", type="primary", disabled=uploaded_file is None, use_container_width=True):
        tmp_path = None
        try:
            with st.spinner("Analyzing Layout & Indexing Context..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
                splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
                chunks = splitter.split_documents(docs)
                texts = [chunk.page_content for chunk in chunks]

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
                        "metadata": {"context": text}
                    })

                for start_idx in range(0, len(vectors), 100):
                    index.upsert(vectors=vectors[start_idx:start_idx + 100], namespace=chat["namespace"])

                chat["pdf_processed"] = True
                chat["doc_name"] = uploaded_file.name
                if chat["name"].startswith("Chat "): chat["name"] = uploaded_file.name[:25]
                st.success("✅ Context ingested into local cache!")
        except Exception as e:
            st.error(f"Ingestion failed: {str(e)}")
        finally:
            if tmp_path and os.path.exists(tmp_path): os.unlink(tmp_path)
        if chat["pdf_processed"]: st.rerun()

    # ========================= INTEGRATED EVALUATION HARNESS =========================
    st.divider()
    st.header("📋 Quality Evaluation")
    eval_mode = st.toggle("Enable Regression Runner")
    if eval_mode:
        eval_questions = [
            {"q": "What is the primary methodology introduced in chapter 1?", "ref": "Methodology description"},
            {"q": "How does the system scale with higher chunk overlap configuration?", "ref": "Cross-chapter synthesis response matching"}
        ]
        
        st.caption("Auto-checks consistency across consecutive loops.")
        if st.button("Run Evaluation Suite", use_container_width=True):
            if not chat["pdf_processed"]:
                st.warning("Ingest a target document first.")
            else:
                test_results = []
                llm_eval = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.0)
                retriever = PineconeHybridSearchRetriever(
                    embeddings=embeddings, sparse_encoder=chat["bm25_encoder"],
                    index=chat["pinecone_index"], alpha=0.5, top_k=6, namespace=chat["namespace"]
                )
                context_builder = AdvancedContextBuilder(reranker)
                
                for i, t_case in enumerate(eval_questions):
                    st.write(f"Evaluating Question {i+1}: *{t_case['q']}*")
                    runs = []
                    for run_idx in range(3):
                        retrieved = retriever.invoke(t_case['q'])
                        compressed_text = context_builder.build_and_compress(retrieved, t_case['q'], max_sentences=15)
                        
                        eval_prompt = f"Using this context, write a brief answer: {compressed_text}\n\nQuestion: {t_case['q']}"
                        ans = llm_eval.invoke(eval_prompt).content
                        runs.append(ans)
                    
                    match = "Pass" if len(set(runs)) == 1 else "Inconsistent"
                    st.write(f"Consistency Status: **{match}**")
                    test_results.append(match)
                
                if "Inconsistent" not in test_results:
                    st.success("🎉 All evaluations run deterministically (100% Match)!")
                else:
                    st.error("⚠️ Inconsistencies detected. Review chunk overlap.")

# ========================= MAIN UI =========================
st.markdown('<p class="main-header">🧠 Advanced Graph RAG System</p>', unsafe_allow_html=True)
st.caption("Multi-Query Synthesis + Persistent Graph + Clean Outputs (Leak-Free)")

chat = st.session_state.chats[st.session_state.current_chat_id]
st.subheader(f"💬 {chat['name']}" + (f"  ·  📄 {chat['doc_name']}" if chat["doc_name"] else ""))

if not chat["pdf_processed"]:
    st.info("👈 Upload a target PDF document in the sidebar, then click **Process Document Context** to begin analysis.")
    st.stop()

for message in chat["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask a complex question requiring cross-chapter analysis...")

if query:
    chat["chat_history"].append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)

    with st.chat_message("assistant"):
        if query.lower().strip() in ["hi", "hello", "hey"]:
            resp = "Greetings. I have analyzed your document. What structured relationships can I explain from its contents?"
            st.markdown(resp)
            chat["chat_history"].append({"role": "assistant", "content": resp})
        else:
            try:
                llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.0)
                retriever = PineconeHybridSearchRetriever(
                    embeddings=embeddings, sparse_encoder=chat["bm25_encoder"],
                    index=chat["pinecone_index"], alpha=0.5, top_k=10, namespace=chat["namespace"]
                )
                kg_rag = KnowledgeGraphRAG(llm, chat["knowledge_graph"])
                context_builder = AdvancedContextBuilder(reranker)

                with st.status("Resolving Entities...", expanded=True) as status:
                    cache_result = chat["semantic_cache"].get_cached_answer(query)

                    if cache_result:
                        response, sim = cache_result
                        status.update(label=f"Answered from Cache (Similarity Match: {sim:.2f})", state="complete")
                    else:
                        status.write("Synthesizing query decomposition nodes (Multi-hop analysis)...")
                        mq_prompt = f"Break down this query into exactly 3 simple logical sub-questions. Do not return intro text. \nQuestion: {query}"
                        sub_queries = llm.invoke(mq_prompt).content.splitlines()
                        sub_queries = [q.strip() for q in sub_queries if q.strip()][:3]
                        sub_queries.append(query)

                        status.write("Executing path retrieval across parallel indices...")
                        all_retrieved = []
                        for sq in sub_queries:
                            all_retrieved.extend(retriever.invoke(sq))

                        unique_docs = {doc.page_content: doc for doc in all_retrieved}
                        retrieved = list(unique_docs.values())

                        if retrieved:
                            status.write("Running Cross-Encoder reranking optimization...")
                            doc_texts = [doc.page_content for doc in retrieved]
                            scores = reranker.predict([[query, text] for text in doc_texts])
                            top_docs = [doc for _, doc in sorted(zip(scores, retrieved), key=lambda x: x[0], reverse=True)[:10]]

                            status.write("Updating Persistent Graph Mapping...")
                            kg_rag.build_graph(top_docs)
                            graph_context = kg_rag.get_graph_context(query)

                            status.write("Constructing context windows...")
                            compressed_text = context_builder.build_and_compress(top_docs, query, max_sentences=25)

                            final_context = ""
                            if graph_context:
                                final_context += graph_context + "\n\n---\n"
                            final_context += "COMPRESSED CONTEXT METADATA:\n" + compressed_text
                        else:
                            final_context = "No relevant context found."

                        status.update(label="Formulating Synthesis...", state="running")

                        final_prompt = f"""You are an elite research analyst. Read the Context Data and formulate a complete answer.

Follow these execution parameters strictly:
1. NO GRAPHICS/LEAKAGE: Never expose raw symbols, arrows (like '->' or '→'), schema structures, or brackets like 'Entity1' in your response. Write everything in clean, fluent prose.
2. ABSOLUTE TRUTHFULNESS: If the provided Context Data does not contain direct proof to support the claim, output this exact sentence: "I don't have enough information in the document to answer that." Do not extrapolate.
3. NO CITATIONS: Do not mention page numbers, source tags, document names, or citation markers in your final answer. 

Context Data:
{final_context}

Question: {query}

Analytical, Citation-Free Response:"""

                        response = llm.invoke(final_prompt).content
                        chat["semantic_cache"].add_to_cache(query, response)
                        status.update(label="Complete", state="complete")

                if cache_result:
                    st.markdown(f"<span class='badge cache-hit'>⚡ CACHE HIT ({sim:.2f})</span><br><br>", unsafe_allow_html=True)

                st.markdown(response)
                chat["chat_history"].append({"role": "assistant", "content": response})

                if not cache_result:
                    with st.expander("🔍 View Synthesis Data (For Portfolio)"):
                        st.markdown("**1. Sub-Queries Generated (Multi-Hop Decomposition):**")
                        for sq in sub_queries:
                            st.write(f"- {sq}")
                        st.markdown("**2. Persistent Graph Relationships (Declarative Formats):**")
                        st.code(graph_context if graph_context else "No active relationships extracted for this context.")
                        st.markdown("**3. Raw Compressed Context:**")
                        st.write(compressed_text)

            except Exception as e:
                st.error(f"Process Error: {str(e)}")
