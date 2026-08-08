import streamlit as st
import os
import time
import tempfile
import uuid
import re
import io as io_module
import numpy as np
import pandas as pd
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

# ========================= OPTIONAL: TABLE / IMAGE EXTRACTION =========================
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ========================= LANGSMITH (TRACING) =========================
try:
    from langsmith import traceable, Client as LangSmithClient
    try:
        from langsmith.run_helpers import get_current_run_tree
    except Exception:
        from langsmith import get_current_run_tree  # fallback for newer SDKs
    LANGSMITH_SDK_AVAILABLE = True
except Exception:
    LANGSMITH_SDK_AVAILABLE = False
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    def get_current_run_tree():
        return None

# ========================= API KEYS =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "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")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY", "pcsk_39EGLB_PC9i9y7MQo2FxSqgqdX4akFP3LPFoNqHirwHsicYqAivgQASB4bFsM9ocPY9epZ")
GROQ_MODEL = os.getenv("GROQ_MODEL") or st.secrets.get("GROQ_MODEL", "llama-3.1-8b-instant")

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY") or st.secrets.get("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT") or st.secrets.get("LANGSMITH_PROJECT", "graph-rag-live-demo")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT") or st.secrets.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

KEYS_CONFIGURED = (
    GROQ_API_KEY and "PASTE_YOUR" not in GROQ_API_KEY and
    PINECONE_API_KEY and "PASTE_YOUR" not in PINECONE_API_KEY
)

LANGSMITH_CONFIGURED = bool(
    LANGSMITH_SDK_AVAILABLE and LANGSMITH_API_KEY and "PASTE_YOUR" not in LANGSMITH_API_KEY
)

# ========================= PAGE CONFIG =========================
st.set_page_config(page_title="Graph RAG • Live Demo", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 42px; font-weight: bold; color: #1E3A8A;}
    .badge {padding: 4px 12px; border-radius: 12px; font-size: 13px; font-weight: bold;}
    .cache-hit {background-color: #22c55e; color: white;}
    .trace-badge {background-color: #6366f1; color: white;}
</style>
""", unsafe_allow_html=True)

if not KEYS_CONFIGURED:
    st.error("⚠️ API keys are not configured yet. Please set `GROQ_API_KEY` and `PINECONE_API_KEY` in the script.")
    st.stop()

# ========================= LANGSMITH SETUP =========================
st.session_state.setdefault("langsmith_tracing_enabled", LANGSMITH_CONFIGURED)

def apply_langsmith_env(enabled: bool):
    if enabled and LANGSMITH_CONFIGURED:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT
        os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

apply_langsmith_env(st.session_state.langsmith_tracing_enabled)

@st.cache_resource
def get_langsmith_client():
    if not LANGSMITH_CONFIGURED:
        return None
    try:
        return LangSmithClient(api_key=LANGSMITH_API_KEY, api_url=LANGSMITH_ENDPOINT)
    except Exception:
        return None

langsmith_client = get_langsmith_client()

def get_trace_url(run_tree):
    if not (run_tree and langsmith_client and st.session_state.langsmith_tracing_enabled):
        return None
    try:
        return langsmith_client.get_run_url(run=run_tree)
    except Exception:
        try:
            return f"https://smith.langchain.com/o/-/projects/p/{LANGSMITH_PROJECT}"
        except Exception:
            return None

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
    cleaned = text
    for pat in LEAK_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()

# ========================= TABLE / IMAGE / CAPTION EXTRACTION HELPERS =========================
CAPTION_REGEX = re.compile(r'((?:Figure|Fig\.?|Table|Chart|Diagram)\s*\d+[:.\-]?\s*[^\n]{0,220})', re.IGNORECASE)

def extract_page_captions(pdf_path):
    """Scan each page's actual text layer for figure/table caption lines
    (e.g. 'Figure 1: AI Agent Market Growth...'). These are usually real,
    extractable text even when the figure itself is an unreadable image, so
    they're the most reliable anchor for answering 'what does Figure N show'
    style questions."""
    captions_by_page = {}
    if not PYMUPDF_AVAILABLE:
        return captions_by_page
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            text = doc[page_num].get_text()
            found = CAPTION_REGEX.findall(text)
            if found:
                captions_by_page[page_num + 1] = [f.strip() for f in found]
        doc.close()
    except Exception:
        pass
    return captions_by_page


def extract_tables_as_markdown(pdf_path):
    """Extract tables with row/column structure preserved (as markdown) so the
    LLM can actually read cells correctly instead of getting jumbled flat text."""
    if not PDFPLUMBER_AVAILABLE:
        return []
    table_chunks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables()
                except Exception:
                    tables = []
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    header = table[0]
                    rows = table[1:]
                    md = "| " + " | ".join(str(h or "").strip() for h in header) + " |\n"
                    md += "| " + " | ".join(["---"] * len(header)) + " |\n"
                    for row in rows:
                        if row is None:
                            continue
                        md += "| " + " | ".join(str(c or "").strip() for c in row) + " |\n"
                    table_chunks.append({
                        "text": f"[TABLE from page {page_num}]\n{md}",
                        "page": page_num,
                        "type": "table",
                    })
    except Exception:
        pass
    return table_chunks


def extract_images_with_ocr(pdf_path, captions_by_page=None):
    """Pull embedded images and (if OCR available) read any text/labels inside
    them. Even when OCR fails or is unavailable, attach any figure/table
    caption found in that page's real text layer — this is the critical fix
    that lets the LLM answer 'what does this figure show' using the caption
    and surrounding described data, instead of refusing outright."""
    if not PYMUPDF_AVAILABLE:
        return []
    captions_by_page = captions_by_page or {}
    image_chunks = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_caps = captions_by_page.get(page_num + 1, [])
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                except Exception:
                    continue

                ocr_text = ""
                if OCR_AVAILABLE:
                    try:
                        pil_img = Image.open(io_module.BytesIO(image_bytes))
                        ocr_text = pytesseract.image_to_string(pil_img).strip()
                    except Exception:
                        ocr_text = ""

                caption_hint = f"[FIGURE/IMAGE on page {page_num + 1}]"
                if ocr_text:
                    caption_hint += f" Extracted text/labels from the image (via OCR): {ocr_text}"
                else:
                    caption_hint += (" No readable text could be extracted directly from the image pixels "
                                      "(it may be a photo, logo, or a chart whose internal labels are too small/stylized for OCR).")

                if page_caps:
                    caption_hint += (" However, the following caption/reference text was found on the SAME PAGE "
                                      "in the document's text layer, and should be used to explain what this "
                                      "figure represents: " + " | ".join(page_caps))

                image_chunks.append({
                    "text": caption_hint,
                    "page": page_num + 1,
                    "type": "image",
                })
        doc.close()
    except Exception:
        pass
    return image_chunks


# ========================= KEYWORD-BOOST RETRIEVAL FOR NAMED FIGURES/TABLES =========================
FIGURE_REF_REGEX = re.compile(r'\b(figure|fig\.?|table|chart|diagram)\s*\.?\s*(\d+)\b', re.IGNORECASE)

def keyword_boost_chunks(query, all_chunks_data, max_matches=8):
    """If the user explicitly names a figure/table/chart number (e.g. 'Figure 1'),
    force-include every chunk in the whole document that literally mentions it —
    captions, references in prose, the image chunk itself, nearby tables — instead
    of relying solely on embedding/BM25 similarity, which often ranks short
    captions too low to surface for vague-sounding questions like 'describe the
    layout of Figure 1'."""
    matches_needed = FIGURE_REF_REGEX.findall(query)
    if not matches_needed:
        return []

    boosted = []
    seen = set()
    for label, num in matches_needed:
        pattern = re.compile(rf'{re.escape(label)}\.?\s*{re.escape(num)}\b', re.IGNORECASE)
        for item in all_chunks_data:
            if pattern.search(item["text"]):
                key = item["text"][:60]
                if key not in seen:
                    seen.add(key)
                    boosted.append(item)
    return boosted[:max_matches]


# ========================= VISUALIZATION HELPERS =========================
VISUAL_KEYWORDS = ["chart", "plot", "visuali", "graph", "trend", "compare", "comparison", "vs", "versus"]

def wants_visualization(query):
    q = query.lower()
    return any(k in q for k in VISUAL_KEYWORDS)

def extract_markdown_table(text):
    """Parse the first markdown table found in the LLM's response into a DataFrame."""
    lines = [l for l in text.splitlines() if l.strip().startswith("|")]
    if len(lines) < 3:
        return None
    try:
        table_str = "\n".join(lines)
        df = pd.read_csv(io_module.StringIO(table_str), sep="|", engine="python", skipinitialspace=True)
        df = df.drop(df.columns[[0, -1]], axis=1)
        df = df.iloc[1:].reset_index(drop=True)  # drop the "---" separator row
        df.columns = [str(c).strip() for c in df.columns]
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.extract(r"([-\d.]+)")[0],
                errors="coerce"
            )
        df = df.dropna(how="all", axis=1)
        return df
    except Exception:
        return None


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
    """Sentence-level dedup + compression for PROSE, with a separate path for
    TABLE and IMAGE chunks (kept intact, ranked as whole blocks — sentence
    tokenization destroys table rows and scores image captions unfairly low)."""
    def __init__(self, cross_encoder):
        self.reranker = cross_encoder

    @traceable(run_type="chain", name="Context Compression & Dedup")
    def build_and_compress(
        self,
        items,
        query,
        max_sentences=22,
        relative_gap=4.0,
        min_sentences_floor=3,
        max_blocks=6,
        debug_scores=False,
    ):
        sentence_candidates = []
        whole_block_candidates = []

        for item in items:
            item_type = item.get("type", "text")
            page = item.get("page", "Unknown")
            if item_type in ("table", "image"):
                whole_block_candidates.append({"text": item["text"], "page": page, "type": item_type})
            else:
                for s in nltk.sent_tokenize(item["text"]):
                    if len(s.strip()) > 20:
                        sentence_candidates.append({"text": s.strip(), "page": page, "type": "text"})

        # ---- Dedup + rank PROSE sentences ----
        unique_sentences, seen = [], set()
        for item in sentence_candidates:
            key = item["text"].lower()
            if key not in seen:
                seen.add(key)
                unique_sentences.append(item)

        score_log = []
        filtered_sentences = []

        if unique_sentences:
            pairs = [[query, item["text"]] for item in unique_sentences]
            scores = self.reranker.predict(pairs)
            ranked = sorted(zip(scores, unique_sentences), key=lambda x: x[0], reverse=True)
            top_k = ranked[:max_sentences]
            score_log = [(float(s), item["text"][:80]) for s, item in top_k] if debug_scores else []

            if top_k:
                best_score = top_k[0][0]
                filtered_sentences = [item for score, item in top_k if score > best_score - relative_gap]
                if not filtered_sentences:
                    filtered_sentences = [item for _, item in top_k[:min_sentences_floor]]

        # ---- Rank TABLE / IMAGE blocks as whole units (never sentence-split) ----
        ranked_blocks = []
        if whole_block_candidates:
            # dedup identical block text first
            seen_blocks = set()
            dedup_blocks = []
            for b in whole_block_candidates:
                key = b["text"][:80]
                if key not in seen_blocks:
                    seen_blocks.add(key)
                    dedup_blocks.append(b)

            block_pairs = [[query, b["text"][:2000]] for b in dedup_blocks]
            block_scores = self.reranker.predict(block_pairs)
            ranked_full = sorted(zip(block_scores, dedup_blocks), key=lambda x: x[0], reverse=True)
            top_blocks = ranked_full[:max_blocks]
            if top_blocks:
                best_block_score = top_blocks[0][0]
                ranked_blocks = [b for s, b in top_blocks if s > best_block_score - relative_gap]
                if not ranked_blocks:
                    ranked_blocks = [b for _, b in top_blocks[:2]]

        if not filtered_sentences and not ranked_blocks:
            return "No relevant context found.", score_log

        parts = []
        if ranked_blocks:
            parts.append("\n\n".join(b["text"] for b in ranked_blocks))
        if filtered_sentences:
            parts.append("\n".join(item["text"] for item in filtered_sentences))

        return "\n\n".join(parts), score_log


# ========================= PARENT/NEIGHBOR EXPANSION =========================
@traceable(run_type="tool", name="Neighbor Expansion (Parent Document)")
def expand_with_neighbors(top_docs, all_chunks_data, window=1):
    """Hierarchical retrieval: pull in the chunk immediately before/after each
    retrieved chunk so connective tissue between two sections isn't cut off mid-thought."""
    selected = {}
    for doc in top_docs:
        idx = doc.metadata.get("chunk_index")
        page = doc.metadata.get("page", "?")
        item_type = doc.metadata.get("type", "text")
        text = doc.page_content
        if idx is None:
            selected[f"raw_{len(selected)}"] = {"text": text, "page": page, "chunk_index": -1, "type": item_type}
            continue
        selected[idx] = {"text": text, "page": page, "chunk_index": idx, "type": item_type}
        for offset in range(1, window + 1):
            for n_idx in (idx - offset, idx + offset):
                if 0 <= n_idx < len(all_chunks_data) and n_idx not in selected:
                    selected[n_idx] = all_chunks_data[n_idx]

    ordered = sorted(selected.values(), key=lambda x: x["chunk_index"] if isinstance(x["chunk_index"], int) else -1)
    return ordered


# ========================= CONSISTENCY SCORING =========================
@traceable(run_type="chain", name="Consistency Scoring")
def compute_consistency(responses):
    if len(responses) < 2:
        return 1.0
    vecs = np.array([embeddings.embed_query(r) for r in responses], dtype=np.float32)
    faiss.normalize_L2(vecs)
    sims = [float(np.dot(vecs[i], vecs[j])) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    return float(np.mean(sims)) if sims else 1.0


# ========================= CORE PIPELINE (SHARED BY CHAT UI + EVAL HARNESS) =========================
@traceable(run_type="chain", name="RAG Pipeline")
def run_rag_pipeline(query, chat, deterministic=True, use_cache=True, status=None):
    def log(msg):
        if status is not None:
            status.write(msg)

    debug = {
        "cache_hit": False, "sub_queries": [], "retrieved_pages": [],
        "compressed_text": "", "final_context": "", "compression_scores": [],
        "trace_url": None, "keyword_boosted": [],
    }

    try:
        run_tree = get_current_run_tree()
        debug["trace_url"] = get_trace_url(run_tree)
    except Exception:
        pass

    run_tags = ["deterministic" if deterministic else "sampled", f"chat:{chat['namespace']}"]
    run_metadata = {"chat_id": chat["namespace"], "doc_name": chat.get("doc_name"), "use_cache": use_cache}

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
multi-part questions spanning different sections, tables, or figures). Output ONLY the
sub-questions, one per line, no numbering, no extra text.
Question: {query}"""
    try:
        raw_sub = llm.invoke(
            mq_prompt,
            config={"tags": run_tags + ["query-decomposition"], "metadata": run_metadata,
                    "run_name": "Query Decomposition"}
        ).content
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
            all_retrieved.extend(
                retriever.invoke(
                    sq,
                    config={"tags": run_tags + ["hybrid-retrieval"], "metadata": run_metadata,
                            "run_name": "Hybrid Retrieval"}
                )
            )
        except Exception:
            continue

    unique_docs = {doc.page_content: doc for doc in all_retrieved}
    retrieved = list(unique_docs.values())

    if not retrieved:
        # Even with zero vector/BM25 hits, a named figure/table might still be
        # findable via literal keyword match — don't give up yet.
        boosted_only = keyword_boost_chunks(query, chat["all_chunks_data"])
        if not boosted_only:
            return "I don't have enough information in the document to answer that.", debug
        expanded_items = boosted_only
        debug["keyword_boosted"] = [b["text"][:120] for b in boosted_only]
    else:
        log("Reranking with cross-encoder...")
        doc_texts = [doc.page_content for doc in retrieved]
        scores = reranker.predict([[query, t] for t in doc_texts])
        top_docs = [d for _, d in sorted(zip(scores, retrieved), key=lambda x: x[0], reverse=True)[:8]]

        debug["retrieved_pages"] = [
            {"page": d.metadata.get("page", "?"), "chunk_index": d.metadata.get("chunk_index", "?"),
             "type": d.metadata.get("type", "text"), "preview": d.page_content[:120] + "..."} for d in top_docs
        ]

        log("Expanding with neighboring context (parent-document retrieval)...")
        expanded_items = expand_with_neighbors(top_docs, chat["all_chunks_data"], window=1)

        # ---- KEYWORD-BOOST: force-include chunks that literally name a
        # figure/table the user asked about, even if ranking missed them ----
        log("Checking for explicitly named figures/tables to force-include...")
        boosted = keyword_boost_chunks(query, chat["all_chunks_data"])
        if boosted:
            existing_keys = {item["text"][:60] for item in expanded_items}
            for b in boosted:
                if b["text"][:60] not in existing_keys:
                    expanded_items.append(b)
                    existing_keys.add(b["text"][:60])
            debug["keyword_boosted"] = [b["text"][:120] for b in boosted]

    log("Compressing & deduplicating context (tables/images kept intact)...")
    context_builder = AdvancedContextBuilder(reranker)
    compressed_text, score_log = context_builder.build_and_compress(
        expanded_items, query, max_sentences=22, debug_scores=(status is not None)
    )
    debug["compressed_text"] = compressed_text
    debug["compression_scores"] = score_log

    if not compressed_text or not compressed_text.strip() or compressed_text == "No relevant context found.":
        log("⚠️ Compression returned empty — falling back to raw top chunks.")
        fallback_texts = [item["text"] for item in expanded_items[:8]]
        compressed_text = "\n".join(fallback_texts) if fallback_texts else "No relevant context found."
        debug["compressed_text"] = compressed_text

    final_context = "SOURCE PASSAGES:\n" + compressed_text
    debug["final_context"] = final_context

    log("Synthesizing final answer...")
    final_prompt = f"""You are a precise, document-grounded analytical assistant used in a
production question-answering system. Your single most important responsibility is
FACTUAL FIDELITY to the provided Context Data. You must never sound confident about
something the Context Data does not actually support — but you must also NEVER refuse
to answer when relevant information genuinely exists somewhere in the Context Data.

The Context Data may contain THREE kinds of content, each marked accordingly:
- Plain prose text (no special marker)
- Tables, marked "[TABLE from page X]" followed by a markdown table
- Figures/images, marked "[FIGURE/IMAGE on page X]" followed by any OCR'd text
  and/or any caption/reference text found elsewhere on that same page

====================================================================
SECTION 1 — CORE GROUNDING PRINCIPLE
====================================================================

- Use ONLY the information explicitly present in the Context Data below.
- Do not use outside knowledge, training data, assumptions, or general world
  knowledge about the topic — even if you "know" the correct answer.
- Do not fill gaps with plausible-sounding information. If the Context Data is
  silent on something, treat it as unknown, not as something you can reasonably guess.
- Numbers, dates, names, and figures must be copied or paraphrased exactly as
  they appear. Never round, estimate, recalculate, or "correct" a number found
  in the Context Data — except where Section 2B explicitly asks you to compute
  a derived value (sum, difference, %, etc.) from stated numbers.

====================================================================
SECTION 2 — CLASSIFY THE QUESTION FIRST (internally, do not show this step)
====================================================================

1. OVERVIEW / SUMMARY → synthesize themes across the entire Context Data.
2. SPECIFIC FACTUAL → answer from the single most relevant passage/table/figure.
3. RELATIONSHIP / CAUSAL → only state a relationship if explicitly described.
4. LIST / ENUMERATION → extract all explicitly stated relevant items only.
5. COMPARISON → only compare attributes explicitly stated for both items.
6. YES/NO or VERIFICATION → answer directly, then support with a paraphrase.
7. MULTI-PART → address each part separately; state which parts aren't covered.

Also apply Section 2B whenever the question involves tables, figures/images,
numeric comparisons, calculations, or visualization requests — these layer on
top of any category above.

====================================================================
SECTION 2B — DATA, TABLE & IMAGE REASONING (perform internally, silently)
====================================================================

A. TABLE-GROUNDED QUESTIONS
   1. Locate every table row/column relevant to the entities in the question.
   2. Extract exact values — never approximate or infer a missing cell.
   3. Compute derived results (difference, sum, %, ranking, trend) yourself
      from extracted values, and SHOW the numbers used
      (e.g., "May: 120 units, June: 150 units → +25%").
   4. If a needed value is missing from every table, say so explicitly.
   5. Reproduce relevant figures as a compact markdown table when the
      question involves 3+ data points or an explicit comparison.

B. IMAGE / FIGURE-GROUNDED QUESTIONS — CRITICAL RULE:
   A figure/image entry with "no readable text detected" from OCR means ONLY
   that the raw pixel content of that specific image could not be read. It
   does NOT mean you lack information about that figure. Before saying you
   cannot answer, you MUST:
     1. Check if a caption or reference to that figure/table number (e.g.
        "Figure 1: ...") appears anywhere in the Context Data — in the image
        entry itself, or in nearby prose/table text — and use it.
     2. Check if the surrounding prose or tables in the Context Data describe
        the same statistics, trends, or subject matter the figure is about
        (this is common — text near a chart usually restates or explains its
        data). If so, use that data to answer the substantive question.
     3. Only state that you cannot help with the parts that GENUINELY require
        seeing the image (e.g., exact colors, exact visual layout/positioning,
        chart type, axis styling) — and clearly separate that limitation from
        the parts of the question you CAN answer using text/caption/table data.
   - Never issue a full blanket refusal to a figure-related question just
     because OCR failed. Partial, clearly-labeled answers are required
     whenever ANY related information exists elsewhere in the Context Data.
   - Never invent visual details (colors, shapes, exact chart type) not
     stated anywhere in the Context Data.

C. VISUALIZATION REQUESTS ("show me a chart", "visualize this", "plot the
   trend", "graph the comparison")
   - Produce the most chart-ready markdown table of the requested data,
     followed by a short written interpretation of the pattern.
   - ALWAYS put the data table BEFORE the interpretation.
   - If requested data spans values not present in the Context Data, build
     the table only from what's available and note what's missing.
   - Never describe a chart in words instead of providing the table.

D. MULTI-STEP ANALYTICAL QUESTIONS (comparisons, trends, calculations
   spanning multiple data points, tables, or sections)
   - Steps: (1) identify required data points, (2) locate each in the
     Context Data, (3) note any missing, (4) compute as needed, (5)
     synthesize a plain-language conclusion.
   - Combining individually-stated values from different tables/pages to
     answer a comparison or calculation IS correct and expected — this is
     NOT the same as blending unrelated prose claims (Section 3).
   - Only say data is unavailable when a specific required number is
     genuinely absent from all retrieved content.

====================================================================
SECTION 3 — ANTI-HALLUCINATION SAFEGUARDS
====================================================================

- Never guess a relationship, cause, or connection not explicitly written
  in the Context Data.
- Never blend two distant, unrelated PROSE claims into one fabricated
  statement (numeric computation across tables per Section 2B-D is exempt).
- If the Context Data is ambiguous or contradictory, report that plainly
  instead of resolving it yourself.
- Do not extrapolate beyond what is directly stated or directly computable.
- If a name, number, or term is not explicitly present anywhere in the
  Context Data, do not introduce it into your answer.

====================================================================
SECTION 4 — HANDLING INSUFFICIENT OR PARTIAL INFORMATION
====================================================================

- Full refusal ("I don't have enough information in the document to answer
  that.") is ONLY appropriate when the Context Data contains ABSOLUTELY
  NOTHING relevant to any part of the question — not even a related caption,
  statistic, or description of the same topic.
- If the Context Data contains SOME relevant information — including
  captions, nearby statistics, or partial table/text data related to a
  figure — you MUST use it. Provide what is supported, and separately and
  explicitly state which specific part of the question remains unanswered
  (e.g., "the document does not let me describe the exact visual layout of
  this chart, but the surrounding text reports the following figures...").
- Never treat "the image itself has no OCR text" as equivalent to "there is
  no relevant information in the Context Data" — check the rest of the
  context first, every time.
- Never pad an incomplete answer with speculation to make it feel complete.

====================================================================
SECTION 5 — OUTPUT FORMATTING RULES
====================================================================

- Write in clear, natural, professional prose.
- NEVER output raw internal formatting artifacts: arrows (→), special bullet
  symbols (•), placeholder labels ("Chunk 1", "Passage A"), internal IDs, or
  other system-level markers.
- Do not include page numbers, section IDs, or citation tags.
- Plain numbered lists ("1.", "2.") are fine; decorative bullet symbols are not.
- EXCEPTION — TABLES ARE REQUIRED, NOT LEAKAGE: when comparing 3+ values,
  numeric trends, or responding to a chart/visualization/comparison request,
  include a clean markdown table of the relevant figures BEFORE your written
  explanation, using standard `| Column | Column |` syntax.

====================================================================
SECTION 6 — TONE AND STYLE
====================================================================

- Be direct and confident when the Context Data clearly supports the answer.
- Be transparent when only partially supported — say "the document states..."
  rather than using absolute certainty language.
- Avoid filler, over-hedging, or repeating the question.
- Match answer length/detail to question complexity.

====================================================================
SECTION 7 — FINAL SELF-CHECK (perform silently before responding)
====================================================================

1. Did I check captions and surrounding text/tables before treating a
   figure/image as unanswerable?
2. Is every claim directly traceable to the Context Data (computations shown)?
3. Have I avoided connecting unrelated prose passages without textual basis?
4. Have I removed all raw symbols, internal labels, citation markers?
5. If something is missing, have I said so explicitly rather than refusing
   the whole question?
6. If tables/figures/comparisons/visualization were involved, did I include
   a markdown table before my explanation?

Only output the final answer — never show this checklist to the user.

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
    raw_response = llm.invoke(
        final_prompt,
        config={"tags": run_tags + ["final-synthesis"], "metadata": run_metadata,
                "run_name": "Final Answer Synthesis"}
    ).content
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
    st.header("📊 Table / Image Extraction")
    if PDFPLUMBER_AVAILABLE:
        st.success("✅ Table extraction enabled (pdfplumber)")
    else:
        st.warning("Table extraction disabled — run `pip install pdfplumber` to enable.")
    if PYMUPDF_AVAILABLE:
        st.success("✅ Image + caption extraction enabled (PyMuPDF)")
    else:
        st.warning("Image extraction disabled — run `pip install pymupdf` to enable.")
    if PYMUPDF_AVAILABLE and not OCR_AVAILABLE:
        st.info("OCR not available — images will still get captions from the text layer, but in-image pixel text won't be read. Run `pip install pytesseract pillow` (+ install Tesseract binary) to enable.")

    st.divider()
    st.header("🔎 LangSmith Tracing")
    if not LANGSMITH_SDK_AVAILABLE:
        st.warning("`langsmith` package not installed. Run `pip install -U langsmith`.")
    elif not LANGSMITH_CONFIGURED:
        st.warning(
            "LangSmith API key not configured. Set `LANGSMITH_API_KEY` (and optionally "
            "`LANGSMITH_PROJECT`) as an env var or Streamlit secret to enable tracing."
        )
    else:
        toggled = st.checkbox(
            "Enable tracing", value=st.session_state.langsmith_tracing_enabled,
            help="Sends every LLM call, retrieval, and pipeline step to LangSmith for inspection."
        )
        if toggled != st.session_state.langsmith_tracing_enabled:
            st.session_state.langsmith_tracing_enabled = toggled
            apply_langsmith_env(toggled)
            st.rerun()

        if st.session_state.langsmith_tracing_enabled:
            st.success(f"✅ Tracing ON · Project: `{LANGSMITH_PROJECT}`")
            st.markdown(f"[🔗 Open LangSmith Project Dashboard](https://smith.langchain.com/)")
        else:
            st.info("Tracing is currently OFF.")

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

                text_items = []
                for c in chunks:
                    text_items.append({
                        "text": c.page_content,
                        "page": c.metadata.get("page", 0) + 1,
                        "type": "text",
                    })

                if PDFPLUMBER_AVAILABLE:
                    st.write("🔧 Extracting tables from PDF...")
                    table_items = extract_tables_as_markdown(tmp_path)
                else:
                    table_items = []

                page_captions = {}
                if PYMUPDF_AVAILABLE:
                    st.write("🔧 Scanning pages for figure/table captions...")
                    page_captions = extract_page_captions(tmp_path)
                    st.write("🔧 Extracting images/figures from PDF...")
                    image_items = extract_images_with_ocr(tmp_path, captions_by_page=page_captions)
                else:
                    image_items = []

                st.write(f"📊 Found {len(table_items)} table(s), {len(image_items)} image/figure(s), "
                         f"and captions on {len(page_captions)} page(s).")

                all_items = text_items + table_items + image_items
                for idx, item in enumerate(all_items):
                    item["chunk_index"] = idx

                texts = [item["text"] for item in all_items]

                st.write("🔧 Fitting BM25 encoder...")
                bm25 = BM25Encoder().default()
                bm25.fit(texts)
                chat["bm25_encoder"] = bm25

                chat["all_chunks_data"] = all_items

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
                for i, item in enumerate(all_items):
                    dense = embeddings.embed_query(item["text"])
                    sparse = bm25.encode_documents([item["text"]])[0]
                    vectors.append({
                        "id": f"chunk_{i}", "values": dense, "sparse_values": sparse,
                        "metadata": {
                            "context": item["text"],
                            "page": item["page"],
                            "chunk_index": item["chunk_index"],
                            "type": item.get("type", "text"),
                            "source": uploaded_file.name
                        }
                    })

                for start_idx in range(0, len(vectors), 100):
                    index.upsert(vectors=vectors[start_idx:start_idx + 100], namespace=chat["namespace"])

                chat["pdf_processed"] = True
                chat["doc_name"] = uploaded_file.name
                if chat["name"].startswith("Chat "):
                    chat["name"] = uploaded_file.name[:30]
                st.success(f"✅ Document processed! {len(all_items)} chunks indexed "
                           f"({len(text_items)} text, {len(table_items)} table, {len(image_items)} image).")

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
st.caption("Deterministic • Multi-Hop Retrieval • Table/Image/Caption Aware • Auto-Visualization • LangSmith Traced")

chat = st.session_state.chats[st.session_state.current_chat_id]
st.subheader(f"💬 {chat['name']}" + (f"  ·  📄 {chat['doc_name']}" if chat["doc_name"] else ""))

if not chat["pdf_processed"]:
    st.info("👈 Upload a PDF for this chat in the sidebar, then click **Process Document** to start chatting.")
    st.stop()

for message in chat["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask any question about your document (text, tables, or figures)...")

if query:
    chat["chat_history"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if query.lower().strip() in ["hi", "hello", "hey"]:
            resp = "Hello! 👋 Ask me anything about your document — including tables, figures, comparisons, or trends — and I'll give you a grounded, detailed answer (with a chart when useful)."
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

                # ---- Auto chart rendering when the user asked for comparison/visualization ----
                if wants_visualization(query):
                    df = extract_markdown_table(response)
                    if df is not None and df.shape[1] >= 2 and df.shape[0] >= 2:
                        try:
                            chart_df = df.set_index(df.columns[0])
                            st.markdown("**📊 Visualization:**")
                            if any(k in query.lower() for k in ["trend", "over time", "timeline"]):
                                st.line_chart(chart_df)
                            else:
                                st.bar_chart(chart_df)
                        except Exception:
                            pass

                if debug.get("trace_url"):
                    st.markdown(
                        f"<span class='badge trace-badge'>🔗 <a href='{debug['trace_url']}' target='_blank' style='color:white;'>View trace in LangSmith</a></span>",
                        unsafe_allow_html=True
                    )

                if st.session_state.show_debug and not debug.get("cache_hit"):
                    with st.expander("🔍 Debug: Retrieval & Context (For QA / Portfolio)"):
                        st.markdown("**Sub-queries used for multi-hop retrieval:**")
                        for sq in debug.get("sub_queries", []):
                            st.write(f"- {sq}")
                        st.markdown("**Retrieved chunks (page / chunk index / type / preview):**")
                        st.json(debug.get("retrieved_pages", []))
                        if debug.get("keyword_boosted"):
                            st.markdown("**Force-included via keyword boost (explicit Figure/Table N reference in query):**")
                            st.json(debug.get("keyword_boosted", []))
                        st.markdown("**Compression scores (score, sentence preview) — for diagnosing empty-context bugs:**")
                        st.json(debug.get("compression_scores", []))
                        st.markdown("**Compressed context sent to the LLM:**")
                        st.write(debug.get("compressed_text", ""))
                        if debug.get("trace_url"):
                            st.markdown(f"**LangSmith trace:** [{debug['trace_url']}]({debug['trace_url']})")

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ========================= EVALUATION HARNESS =========================
st.divider()
with st.expander("🧪 Evaluation Harness — Consistency Testing"):
    st.caption(
        "Run the same questions multiple times to verify determinism and detect drift — "
        "without doing it manually every time. Every run is also traced to LangSmith "
        "(if enabled) under the `deterministic` / `sampled` tags for later inspection."
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
            trace_urls = []
            for _ in range(runs_per_q):
                resp, dbg = run_rag_pipeline(q, chat, deterministic=True, use_cache=False, status=None)
                responses.append(resp)
                trace_urls.append(dbg.get("trace_url"))
                step += 1
                progress.progress(step / total)

            consistency = compute_consistency(responses)
            results.append({
                "Question": q,
                "Consistency Score (0-1)": round(consistency, 3),
                "Sample Answer": responses[0][:200] + ("..." if len(responses[0]) > 200 else ""),
                "Trace (Run 1)": trace_urls[0] if trace_urls else None,
            })

        st.session_state[f"eval_results_{st.session_state.current_chat_id}"] = results

    results_key = f"eval_results_{st.session_state.current_chat_id}"
    if results_key in st.session_state:
        st.dataframe(st.session_state[results_key], use_container_width=True)
        st.caption(
            "Consistency Score ≥ 0.9 typically means near-identical answers across runs. "
            "Anything below ~0.7 indicates the pipeline is still non-deterministic for that question type. "
            "Click a trace link to inspect the exact retrieval/context/prompt for that specific run in LangSmith."
        )
