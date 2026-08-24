import streamlit as st
import os
import time
import tempfile
import uuid
import re
import io as io_module
import base64
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
from langchain_core.messages import HumanMessage
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
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ========================= LANGSMITH (TRACING) =========================
try:
    from langsmith import traceable, Client as LangSmithClient
    try:
        from langsmith.run_helpers import get_current_run_tree
    except Exception:
        from langsmith import get_current_run_tree
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
GROQ_MODEL = os.getenv("GROQ_MODEL") or st.secrets.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# Vision-capable Groq models (fallback list)
DEFAULT_VISION_MODELS = [
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview",
]

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY") or st.secrets.get("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT") or st.secrets.get("LANGSMITH_PROJECT", "graph-rag-live-demo")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT") or st.secrets.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

KEYS_CONFIGURED = (
    GROQ_API_KEY and "PASTE_YOUR" not in GROQ_API_KEY and
    PINECONE_API_KEY and "PASTE_YOUR" not in PINECONE_API_KEY
)

VISION_CONFIGURED = bool(KEYS_CONFIGURED)

LANGSMITH_CONFIGURED = bool(
    LANGSMITH_SDK_AVAILABLE and LANGSMITH_API_KEY and "PASTE_YOUR" not in LANGSMITH_API_KEY
)

# ========================= PAGE CONFIG & CUSTOM CSS =========================
st.set_page_config(page_title="Neural RAG", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    /* Hide Streamlit default header, footer, and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Global font & background */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background: linear-gradient(160deg, #f8fafc 0%, #e2e8f0 100%);
    }

    /* Main header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }

    /* Chat message styling */
    div[data-testid="stChatMessage"] {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #e2e8f0;
    }
    /* Differentiate user vs assistant bubbles subtly */
    div[data-testid="stChatMessage"]:has(.stChatMessageAvatarUser) {
        background-color: #f0f4ff;
        border-color: #c7d2fe;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .cache-hit { background-color: #dcfce7; color: #166534; }
    .vision-badge { background-color: #fef3c7; color: #92400e; }
    .trace-badge { background-color: #e0e7ff; color: #3730a3; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #334155;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        border-radius: 12px;
        border: 1px solid #cbd5e1;
    }
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

# ========================= LLM FACTORIES =========================
def get_llm(deterministic=True):
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.0 if deterministic else 0.7,
    )

def get_vision_llm(model_name):
    """Return a ChatGroq instance for a specific vision model."""
    return ChatGroq(
        model=model_name,
        api_key=GROQ_API_KEY,
        temperature=0.0,
    )

# ========================= VISION HELPER: TRY MULTIPLE MODELS =========================
def try_vision_call(prompt_text, image_b64, image_ext, model_list):
    """Try a list of vision models in order. Returns (result, error_message)."""
    errors = []
    for model in model_list:
        try:
            llm_vision = get_vision_llm(model)
            message = HumanMessage(content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/{image_ext};base64,{image_b64}"}},
            ])
            result = llm_vision.invoke([message])
            if result and result.content:
                return result.content.strip(), None
            else:
                errors.append(f"Model {model}: returned empty content.")
        except Exception as e:
            errors.append(f"Model {model}: {str(e)}")
    return None, "\n".join(errors)

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

def safe_int(x, default=None):
    try:
        return int(round(float(x)))
    except Exception:
        return default

# ========================= IMAGE HELPERS (RESIZE / ENCODE) =========================
def resize_image_for_vision(image_bytes, max_dim=1024):
    """Downscale large images before sending to the vision model."""
    if not PIL_AVAILABLE:
        return image_bytes, "png"
    try:
        img = Image.open(io_module.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        scale = min(1.0, max_dim / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)))
        buf = io_module.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), "png"
    except Exception:
        return image_bytes, "png"

def encode_image_b64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

# ========================= VISION MODEL CALLS (with fallback) =========================
def describe_image_generic(image_b64, image_ext, page, caption_hint="", model_list=None):
    """INGESTION-TIME: Describe image using fallback vision models."""
    if model_list is None:
        model_list = st.session_state.get("vision_model_list", DEFAULT_VISION_MODELS)
    prompt_text = f"""Describe this image from page {page} of a document in full, factual
detail for someone who cannot see it. Include: the type of visual (bar chart, line chart,
pie chart, table-like image, diagram, photo, logo, etc.), any title, axis labels, legend
entries, every visible numeric data point and what it corresponds to, and any other text
printed in the image. Be precise with numbers — copy them exactly as shown, do not round
or estimate. If this is not a data visualization (e.g., a logo or decorative photo), say
so briefly instead of inventing data.
{"A caption associated with this image in the document text reads: " + caption_hint if caption_hint else ""}"""
    return try_vision_call(prompt_text, image_b64, image_ext, model_list)

def analyze_image_for_question(image_b64, image_ext, page, user_question, caption_hint="", model_list=None):
    """QUERY-TIME: Re-examine image with the user's specific question, using fallback models."""
    if model_list is None:
        model_list = st.session_state.get("vision_model_list", DEFAULT_VISION_MODELS)
    prompt_text = f"""You are looking directly at an image extracted from page {page} of a
document. Answer the user's question using ONLY what is visually present in this image —
chart type, title, axis labels, legend, every visible numeric data point, trend lines,
layout/positioning, colors (if relevant to distinguishing categories), and any printed text.
Be precise with numbers; do not round or estimate. If the image does not contain information
relevant to the question, say so plainly instead of guessing.
{"A caption/reference associated with this image reads: " + caption_hint if caption_hint else ""}

User question: {user_question}"""
    return try_vision_call(prompt_text, image_b64, image_ext, model_list)

# ========================= IMAGE QUERY DETECTION =========================
IMAGE_QUERY_KEYWORDS = ["figure", "fig.", "chart", "image", "diagram", "graph", "visual", "picture", "plot", "layout", "table"]
FIGURE_REF_REGEX = re.compile(r'\b(figure|fig\.?|table|chart|diagram)\s*\.?\s*(\d+)\b', re.IGNORECASE)

def query_mentions_visual(query):
    q = query.lower()
    return any(k in q for k in IMAGE_QUERY_KEYWORDS) or bool(FIGURE_REF_REGEX.search(query))

# ========================= TABLE / CAPTION EXTRACTION HELPERS =========================
CAPTION_REGEX = re.compile(r'((?:Figure|Fig\.?|Table|Chart|Diagram)\s*\d+[:.\-]?\s*[^\n]{0,220})', re.IGNORECASE)

def extract_page_captions(pdf_path):
    """Scan each page's real text layer for figure/table caption lines."""
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
    """Extract tables with row/column structure preserved (as markdown)."""
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

# ========================= IMAGE EXTRACTION + VISION DESCRIPTION (INGESTION) =========================
def extract_and_describe_images(pdf_path, captions_by_page=None, use_vision=True,
                                 max_vision_images=10, progress_cb=None):
    """Full ingestion-time image pipeline with fallback and error capture."""
    if not PYMUPDF_AVAILABLE:
        return []
    captions_by_page = captions_by_page or {}
    image_chunks = []
    vision_calls_used = 0

    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_caps = captions_by_page.get(page_num + 1, [])
            caption_text = " | ".join(page_caps)

            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                except Exception:
                    continue

                # Skip tiny images (icons/bullets/decorative dividers)
                if len(image_bytes) < 3000:
                    continue

                resized_bytes, ext = resize_image_for_vision(image_bytes)
                b64 = encode_image_b64(resized_bytes)

                description = None
                vision_error = None
                if use_vision and VISION_CONFIGURED and vision_calls_used < max_vision_images:
                    if progress_cb:
                        progress_cb(f"🖼️ Vision model analyzing image on page {page_num + 1}...")
                    description, vision_error = describe_image_generic(b64, ext, page_num + 1, caption_text)
                    vision_calls_used += 1

                ocr_text = ""
                if not description and OCR_AVAILABLE:
                    try:
                        pil_img = Image.open(io_module.BytesIO(image_bytes))
                        ocr_text = pytesseract.image_to_string(pil_img).strip()
                    except Exception:
                        ocr_text = ""

                if description:
                    text_block = f"[FIGURE/IMAGE on page {page_num + 1} — described via direct visual analysis]\n{description}"
                elif ocr_text:
                    text_block = f"[FIGURE/IMAGE on page {page_num + 1}] Extracted text/labels via OCR: {ocr_text}"
                else:
                    text_block = (f"[FIGURE/IMAGE on page {page_num + 1}] No readable text or vision "
                                  f"analysis could be extracted from this image.")

                if caption_text:
                    text_block += f"\nCaption/reference found on this page: {caption_text}"

                image_chunks.append({
                    "text": text_block,
                    "page": page_num + 1,
                    "type": "image",
                    "image_b64": b64,
                    "image_ext": ext,
                    "caption_text": caption_text,
                    "vision_described": bool(description),
                    "vision_error": vision_error,   # store error if any
                })
        doc.close()
    except Exception:
        pass
    return image_chunks

# ========================= KEYWORD-BOOST RETRIEVAL =========================
def keyword_boost_chunks(query, all_chunks_data, max_matches=8):
    """If the user explicitly names a figure/table/chart number, force-include every chunk that mentions it."""
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

def boost_image_chunks(query, all_chunks_data, max_images=3):
    """If the query mentions any visual term, add all image chunks to ensure they are considered."""
    q = query.lower()
    if not any(k in q for k in IMAGE_QUERY_KEYWORDS):
        return []

    image_chunks = [item for item in all_chunks_data if item.get("type") == "image" and item.get("image_b64")]
    image_chunks.sort(key=lambda x: x.get("page", 0))
    return image_chunks[:max_images]

# ========================= VISUALIZATION (CHART RENDERING) HELPERS =========================
VISUAL_KEYWORDS = ["chart", "plot", "visuali", "graph", "trend", "compare", "comparison", "vs", "versus"]

def wants_visualization(query):
    q = query.lower()
    return any(k in q for k in VISUAL_KEYWORDS)

def extract_markdown_table(text):
    lines = [l for l in text.splitlines() if l.strip().startswith("|")]
    if len(lines) < 3:
        return None
    try:
        table_str = "\n".join(lines)
        df = pd.read_csv(io_module.StringIO(table_str), sep="|", engine="python", skipinitialspace=True)
        df = df.drop(df.columns[[0, -1]], axis=1)
        df = df.iloc[1:].reset_index(drop=True)
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
    """Sentence-level dedup + compression for PROSE, with a separate path for TABLE and IMAGE chunks."""
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

        ranked_blocks = []
        if whole_block_candidates:
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
    """Pulls in adjacent chunks for context continuity. Uses FULL local record."""
    selected = {}
    for doc in top_docs:
        raw_idx = doc.metadata.get("chunk_index")
        idx = safe_int(raw_idx)
        page = doc.metadata.get("page", "?")
        item_type = doc.metadata.get("type", "text")
        text = doc.page_content

        if idx is not None and 0 <= idx < len(all_chunks_data):
            selected[idx] = all_chunks_data[idx]
        elif idx is not None:
            selected[idx] = {"text": text, "page": page, "chunk_index": idx, "type": item_type}
        else:
            selected[f"raw_{len(selected)}"] = {"text": text, "page": page, "chunk_index": -1, "type": item_type}

        if idx is not None:
            for offset in range(1, window + 1):
                for n_idx in (idx - offset, idx + offset):
                    if 0 <= n_idx < len(all_chunks_data) and n_idx not in selected:
                        selected[n_idx] = all_chunks_data[n_idx]

    ordered = sorted(
        selected.values(),
        key=lambda x: x["chunk_index"] if isinstance(x.get("chunk_index"), int) else -1
    )
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


# ========================= CORE PIPELINE =========================
@traceable(run_type="chain", name="RAG Pipeline")
def run_rag_pipeline(query, chat, deterministic=True, use_cache=True, status=None):
    def log(msg):
        if status is not None:
            status.write(msg)

    debug = {
        "cache_hit": False, "sub_queries": [], "retrieved_pages": [],
        "compressed_text": "", "final_context": "", "compression_scores": [],
        "trace_url": None, "keyword_boosted": [], "vision_analyses": [],
        "image_display": [], "vision_errors": [], "image_boosted": [],
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
        # Still try to boost image chunks if visual query
        boosted_only = keyword_boost_chunks(query, chat["all_chunks_data"])
        if not boosted_only and query_mentions_visual(query):
            boosted_only = boost_image_chunks(query, chat["all_chunks_data"])
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

        log("Checking for explicitly named figures/tables to force-include...")
        boosted = keyword_boost_chunks(query, chat["all_chunks_data"])
        if boosted:
            existing_keys = {item["text"][:60] for item in expanded_items}
            for b in boosted:
                if b["text"][:60] not in existing_keys:
                    expanded_items.append(b)
                    existing_keys.add(b["text"][:60])
            debug["keyword_boosted"] = [b["text"][:120] for b in boosted]

        # NEW: For any visual query, add image chunks directly
        if query_mentions_visual(query):
            image_boost = boost_image_chunks(query, chat["all_chunks_data"])
            existing_keys = {item["text"][:60] for item in expanded_items}
            for img in image_boost:
                key = img["text"][:60]
                if key not in existing_keys:
                    expanded_items.append(img)
                    existing_keys.add(key)
            debug["image_boosted"] = [img["text"][:120] for img in image_boost]

    # ---- QUERY-TIME TARGETED VISION RE-ANALYSIS ----
    vision_context_blocks = []
    if VISION_CONFIGURED and st.session_state.get("enable_vision", True) and query_mentions_visual(query):
        image_candidates = [it for it in expanded_items if it.get("type") == "image" and it.get("image_b64")]

        # Store image display info for inline viewing
        debug["image_display"] = [
            {"page": img.get("page"), "image_b64": img["image_b64"], "caption_text": img.get("caption_text", "")}
            for img in image_candidates[:2]
        ]

        for img_item in image_candidates[:3]:
            log(f"👁️ Running targeted vision analysis on figure (page {img_item.get('page')})...")
            analysis, error = analyze_image_for_question(
                img_item["image_b64"], img_item.get("image_ext", "png"),
                img_item.get("page"), query, img_item.get("caption_text", "")
            )
            if analysis:
                vision_context_blocks.append(
                    f"[VISION MODEL ANALYSIS of figure/image on page {img_item.get('page')} — "
                    f"generated by directly viewing this image to answer the current question]\n{analysis}"
                )
            else:
                debug["vision_errors"].append(f"Vision analysis error for page {img_item.get('page')}: {error}")
    debug["vision_analyses"] = vision_context_blocks

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

    context_pieces = []
    if vision_context_blocks:
        context_pieces.append("\n\n".join(vision_context_blocks))
    context_pieces.append(compressed_text)
    final_context = "SOURCE PASSAGES:\n" + "\n\n".join(context_pieces)
    debug["final_context"] = final_context

    log("Synthesizing final answer...")
    final_prompt = f"""You are a precise, document-grounded analytical assistant used in a
production question-answering system. Your single most important responsibility is
FACTUAL FIDELITY to the provided Context Data. You must never sound confident about
something the Context Data does not actually support — but you must also NEVER refuse
to answer when relevant information genuinely exists somewhere in the Context Data.

The Context Data may contain FOUR kinds of content, each marked accordingly:
- Plain prose text (no special marker)
- Tables, marked "[TABLE from page X]" followed by a markdown table
- Figure/image descriptions generated at ingestion time, marked
  "[FIGURE/IMAGE on page X — described via direct visual analysis]" (or, if
  vision analysis wasn't available for that image, OCR text or a plain
  "[FIGURE/IMAGE on page X]" marker with only a caption)
- Vision model analyses generated specifically for THIS question, marked
  "[VISION MODEL ANALYSIS of figure/image on page X — generated by directly
  viewing this image to answer the current question]" — this is the model
  literally looking at the image right now to answer what was asked.

====================================================================
SECTION 1 — CORE GROUNDING PRINCIPLE
====================================================================

- Use ONLY the information explicitly present in the Context Data below.
- Do not use outside knowledge, training data, assumptions, or general world
  knowledge about the topic — even if you "know" the correct answer.
- Do not fill gaps with plausible-sounding information.
- Numbers, dates, names, and figures must be copied or paraphrased exactly as
  they appear. Never round, estimate, recalculate, or "correct" a number —
  except where Section 2B explicitly asks you to compute a derived value.

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
8. DEFINITION / EXPLANATION → give a definition or explanation based on Context Data.
9. MATH / CALCULATION → if the question asks for arithmetic, compute from numbers
   found in the Context Data and show the steps.
10. CODE / EXTRACTION → if the question asks for code, reproduce it exactly as it
   appears in the Context Data; do not modify or reformat.

Also apply Section 2B whenever the question involves tables, figures/images,
numeric comparisons, calculations, or visualization requests.

====================================================================
SECTION 2B — DATA, TABLE & IMAGE REASONING (perform internally, silently)
====================================================================

A. TABLE-GROUNDED QUESTIONS
   1. Locate every table row/column relevant to the entities in the question.
   2. Extract exact values — never approximate or infer a missing cell.
   3. Compute derived results (difference, sum, %, ranking, trend) yourself
      from extracted values, and SHOW the numbers used.
   4. If a needed value is missing from every table, say so explicitly.
   5. Reproduce relevant figures as a compact markdown table when the
      question involves 3+ data points or an explicit comparison.

B. IMAGE / FIGURE-GROUNDED QUESTIONS — CRITICAL RULES:
   - If a "[VISION MODEL ANALYSIS ...]" block is present for the figure being
     asked about, treat it as AUTHORITATIVE, DIRECT VISUAL OBSERVATION. You
     may describe its layout, chart type, axis labels, trends, and every
     number it reports as established fact — there is no need to hedge about
     "not being able to see the image," because the vision analysis IS you
     having looked at it. Answer fully and confidently using that block.
   - If NO vision analysis block is present for the relevant figure, but a
     "[FIGURE/IMAGE ... described via direct visual analysis]" block from
     ingestion exists, use that description the same way — it also came from
     directly viewing the image.
   - Only fall back to a limitation statement (e.g., "the exact visual layout
     of this image isn't available") when NEITHER a vision analysis block NOR
     a vision-described ingestion block exists for that specific figure —
     i.e., only plain OCR text or a "no text detected" placeholder is present.
     Even then, still use any caption or surrounding prose/table data that
     relates to the same figure before saying anything is unanswerable.
   - Never invent visual details not stated in any of these blocks.

C. VISUALIZATION REQUESTS ("show me a chart", "visualize this", "plot the
   trend", "graph the comparison")
   - Produce the most chart-ready markdown table of the requested data,
     followed by a short written interpretation of the pattern.
   - ALWAYS put the data table BEFORE the interpretation.
   - If requested data spans values not present in the Context Data, build
     the table only from what's available and note what's missing.

D. MULTI-STEP ANALYTICAL QUESTIONS (comparisons, trends, calculations
   spanning multiple data points, tables, figures, or sections)
   - Steps: (1) identify required data points, (2) locate each in the
     Context Data (including vision analysis blocks), (3) note any missing,
     (4) compute as needed, (5) synthesize a plain-language conclusion.
   - Combining individually-stated values from different tables/figures/pages
     to answer a comparison or calculation IS correct and expected.
   - Only say data is unavailable when a specific required number is
     genuinely absent from all retrieved content, including vision blocks.

====================================================================
SECTION 3 — ANTI-HALLUCINATION SAFEGUARDS
====================================================================

- Never guess a relationship, cause, or connection not explicitly written
  in the Context Data.
- Never blend two distant, unrelated PROSE claims into one fabricated
  statement (numeric computation across tables/figures per Section 2B-D is exempt).
- If the Context Data is ambiguous or contradictory, report that plainly.
- Do not extrapolate beyond what is directly stated or directly computable.
- If a name, number, or term is not explicitly present anywhere in the
  Context Data (including vision blocks), do not introduce it.

====================================================================
SECTION 4 — HANDLING INSUFFICIENT OR PARTIAL INFORMATION
====================================================================

- Full refusal ("I don't have enough information in the document to answer
  that.") is ONLY appropriate when the Context Data contains ABSOLUTELY
  NOTHING relevant — not even a related caption, vision analysis, statistic,
  or description of the same topic.
- If the Context Data contains SOME relevant information, use it. Provide
  what is supported, and explicitly state which specific part remains
  unanswered.
- Never treat "OCR found no text" as equivalent to "no information exists" —
  check for vision analysis blocks and surrounding context first, every time.
- Never pad an incomplete answer with speculation.

====================================================================
SECTION 5 — OUTPUT FORMATTING RULES
====================================================================

- Write in clear, natural, professional prose.
- NEVER output raw internal formatting artifacts: arrows (→), special bullet
  symbols (•), placeholder labels ("Chunk 1", "Passage A"), internal IDs.
- Do not include page numbers, section IDs, or citation tags.
- Plain numbered lists ("1.", "2.") are fine; decorative bullet symbols are not.
- EXCEPTION — TABLES ARE REQUIRED, NOT LEAKAGE: when comparing 3+ values,
  numeric trends, or responding to a chart/visualization/comparison request,
  include a clean markdown table of the relevant figures BEFORE your written
  explanation, using standard `| Column | Column |` syntax.

====================================================================
SECTION 6 — TONE AND STYLE
====================================================================

- Be direct and confident when the Context Data (including vision analysis)
  clearly supports the answer.
- Be transparent when only partially supported.
- Avoid filler, over-hedging, or repeating the question.
- Match answer length/detail to question complexity.

====================================================================
SECTION 7 — FINAL SELF-CHECK (perform silently before responding)
====================================================================

1. Did I check for a vision analysis block before treating a figure as
   unanswerable?
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
st.session_state.setdefault("enable_vision", True)
st.session_state.setdefault("vision_model_list", DEFAULT_VISION_MODELS)
st.session_state.setdefault("max_vision_images", 10)

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
    st.markdown("<h2 style='margin-bottom: 0.5rem;'>💬 Chats</h2>", unsafe_allow_html=True)
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
    st.markdown("<h2>⚙️ Settings</h2>", unsafe_allow_html=True)
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
    st.markdown("<h2>🖼️ Vision Model</h2>", unsafe_allow_html=True)
    if PYMUPDF_AVAILABLE and VISION_CONFIGURED:
        st.success("✅ Vision pipeline available")
    elif not PYMUPDF_AVAILABLE:
        st.warning("Install `pymupdf` to enable image extraction: `pip install pymupdf`")
    else:
        st.warning("Vision requires a configured GROQ_API_KEY.")

    st.session_state.enable_vision = st.checkbox(
        "Enable vision-based image understanding",
        value=st.session_state.enable_vision,
        help="Uses a multimodal Groq model to actually look at extracted images/figures."
    )
    with st.expander("Advanced vision settings"):
        # Let user edit the model list (one per line)
        current_models = "\n".join(st.session_state.vision_model_list)
        new_models = st.text_area(
            "Vision model names (one per line, tried in order)",
            value=current_models,
            height=100,
            key="vision_model_list_editor"
        )
        if new_models != current_models:
            st.session_state.vision_model_list = [m.strip() for m in new_models.splitlines() if m.strip()]

        st.session_state.max_vision_images = st.slider(
            "Max images to analyze with vision at ingestion", 1, 25, st.session_state.max_vision_images,
            help="Caps vision API calls during document processing to control cost/time."
        )

        # Test button
        if st.button("🔍 Test Vision Model", use_container_width=True):
            with st.spinner("Testing vision model..."):
                # Create a simple test image (a colored square with text)
                if PIL_AVAILABLE:
                    img = Image.new('RGB', (200, 100), color=(73, 109, 137))
                    draw = ImageDraw.Draw(img)
                    draw.text((10, 40), "Hello Vision", fill=(255, 255, 0))
                    buf = io_module.BytesIO()
                    img.save(buf, format="PNG")
                    test_img_b64 = encode_image_b64(buf.getvalue())
                    test_ext = "png"
                    test_prompt = "Describe what you see in this image."
                    result, err = try_vision_call(test_prompt, test_img_b64, test_ext, st.session_state.vision_model_list)
                    if result:
                        st.success("✅ Vision model works! Response:")
                        st.write(result)
                    else:
                        st.error("❌ Vision test failed. Error(s):")
                        st.code(err)
                else:
                    st.error("Pillow not installed, cannot create test image.")

    if PYMUPDF_AVAILABLE and not OCR_AVAILABLE:
        st.info("OCR fallback not installed — that's fine as long as vision is enabled.")
    if PDFPLUMBER_AVAILABLE:
        st.success("✅ Table extraction enabled (pdfplumber)")
    else:
        st.warning("Table extraction disabled — run `pip install pdfplumber` to enable.")

    st.divider()
    st.markdown("<h2>🔎 LangSmith Tracing</h2>", unsafe_allow_html=True)
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
    st.markdown("<h2>🛠️ Document Setup</h2>", unsafe_allow_html=True)

    new_name = st.text_input("Chat name", value=chat["name"], key=f"name_{st.session_state.current_chat_id}")
    if new_name and new_name != chat["name"]:
        chat["name"] = new_name

    uploaded_file = st.file_uploader("Upload PDF", type="pdf", key=f"upload_{st.session_state.current_chat_id}")

    if st.button("Process Document", type="primary", disabled=uploaded_file is None):
        tmp_path = None
        try:
            with st.spinner("Processing document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                if not docs:
                    raise ValueError("No content could be extracted from the PDF.")

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)

                text_items = [
                    {"text": c.page_content, "page": c.metadata.get("page", 0) + 1, "type": "text"}
                    for c in chunks
                ]

                if PDFPLUMBER_AVAILABLE:
                    table_items = extract_tables_as_markdown(tmp_path)
                else:
                    table_items = []

                page_captions = {}
                image_items = []
                if PYMUPDF_AVAILABLE:
                    page_captions = extract_page_captions(tmp_path)
                    image_items = extract_and_describe_images(
                        tmp_path,
                        captions_by_page=page_captions,
                        use_vision=st.session_state.enable_vision,
                        max_vision_images=st.session_state.max_vision_images,
                        progress_cb=None,  # silent
                    )

                vision_count = sum(1 for it in image_items if it.get("vision_described"))
                vision_errors_during_ingest = [it.get("vision_error") for it in image_items if it.get("vision_error")]

                all_items = text_items + table_items + image_items
                for idx, item in enumerate(all_items):
                    item["chunk_index"] = idx

                texts = [item["text"] for item in all_items]

                bm25 = BM25Encoder().default()
                bm25.fit(texts)
                chat["bm25_encoder"] = bm25

                chat["all_chunks_data"] = all_items

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
                if vision_errors_during_ingest:
                    st.warning("Some vision descriptions failed during ingestion. See debug info for details.")
                    if st.session_state.show_debug:
                        for err in vision_errors_during_ingest[:3]:
                            st.code(err)

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
st.markdown('<p class="main-header">🧠 Neural RAG</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Deterministic · Multi-Hop Retrieval · Vision-Grounded Tables & Figures · Auto-Visualization</p>', unsafe_allow_html=True)

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
            resp = "Hello! 👋 Ask me anything about your document — including tables, figures, comparisons, or trends — and I'll give you a grounded, detailed answer (with vision analysis and charts when useful)."
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
                if debug.get("vision_analyses"):
                    st.markdown(
                        "<span class='badge vision-badge'>👁️ Vision model analyzed an image for this answer</span><br><br>",
                        unsafe_allow_html=True
                    )

                st.markdown(response)
                chat["chat_history"].append({"role": "assistant", "content": response})

                # Show relevant images inline
                if debug.get("image_display"):
                    st.markdown("**🖼️ Relevant image(s) from the document:**")
                    cols = st.columns(min(len(debug["image_display"]), 3))
                    for idx, img_info in enumerate(debug["image_display"]):
                        with cols[idx % 3]:
                            try:
                                img_bytes = base64.b64decode(img_info["image_b64"])
                                st.image(
                                    img_bytes,
                                    caption=f"Page {img_info['page']} — {img_info.get('caption_text','')[:100]}",
                                    use_container_width=True
                                )
                            except Exception:
                                pass

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
                            st.markdown("**Force-included via keyword boost (explicit Figure/Table N reference):**")
                            st.json(debug.get("keyword_boosted", []))
                        if debug.get("image_boosted"):
                            st.markdown("**Force-included image chunks (visual query detected):**")
                            st.json(debug.get("image_boosted", []))
                        if debug.get("vision_analyses"):
                            st.markdown("**🖼️ Live vision model analysis performed for this question:**")
                            for va in debug["vision_analyses"]:
                                st.text(va)
                        if debug.get("vision_errors"):
                            st.markdown("**⚠️ Vision errors encountered:**")
                            for err in debug["vision_errors"]:
                                st.error(err)
                        st.markdown("**Compression scores (score, sentence preview):**")
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
        "Run the same questions multiple times to verify determinism and detect drift. "
        "Every run is also traced to LangSmith (if enabled) under `deterministic`/`sampled` tags."
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
