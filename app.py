"""
app.py
------
Main Streamlit application for the RAG Chatbot.

UI Flow:
  1. User uploads a PDF in the sidebar.
  2. App processes the PDF: loads → splits → embeds → stores in FAISS.
  3. User types questions in the chat input.
  4. App queries the RAG chain and displays the answer + source references.
  5. Full chat history is shown in the main area.
"""

import streamlit as st
from dotenv import load_dotenv

from utils.pdf_loader import load_pdf_from_upload, get_document_metadata
from utils.text_splitter import split_documents, get_chunk_stats
from utils.embeddings import get_embedding_model
from utils.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
    vector_store_exists,
)
from utils.rag_chain import build_rag_chain, query_rag_chain
from utils.llm_provider import (
    build_llm,
    AVAILABLE_PROVIDERS,
    OLLAMA_MODELS,
    DEFAULT_CLOUD_MODELS,
)

# Load .env variables
load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Main background */
        .stApp { background-color: #0f1117; color: #e0e0e0; }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #1a1d27;
            border-right: 1px solid #2e3250;
        }

        /* Chat message bubbles */
        .user-bubble {
            background: #1e3a5f;
            border-radius: 12px 12px 2px 12px;
            padding: 12px 16px;
            margin: 8px 0;
            max-width: 80%;
            margin-left: auto;
            color: #dce8f5;
        }
        .assistant-bubble {
            background: #1c2333;
            border: 1px solid #2e3250;
            border-radius: 12px 12px 12px 2px;
            padding: 12px 16px;
            margin: 8px 0;
            max-width: 85%;
            color: #e0e0e0;
        }

        /* Source expander */
        .source-box {
            background: #141720;
            border-left: 3px solid #4a6fa5;
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 0 6px 6px 0;
            font-size: 0.82em;
            color: #9ab0cc;
        }

        /* Status badge */
        .badge-green {
            background: #1a3a2a;
            color: #4caf7d;
            border: 1px solid #2d6e4a;
            border-radius: 20px;
            padding: 2px 10px;
            font-size: 0.78em;
        }
        .badge-grey {
            background: #252836;
            color: #7a8099;
            border: 1px solid #3a3f5c;
            border-radius: 20px;
            padding: 2px 10px;
            font-size: 0.78em;
        }

        /* Hide Streamlit branding */
        #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session State Initialisation ──────────────────────────────────────────────
# All persistent state lives here; Streamlit reruns this file on every interaction.

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of {"role": ..., "content": ...}

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None  # The built ConversationalRetrievalChain

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None  # Loaded FAISS vector store

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False  # Whether a PDF has been processed

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""  # Filename of the processed PDF

if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}  # Metadata shown in the sidebar

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None  # Cached embedding model (avoid reload)

if "provider" not in st.session_state:
    st.session_state.provider = "Ollama"  # Selected answering provider
if "model" not in st.session_state:
    st.session_state.model = OLLAMA_MODELS[0]  # Selected model for that provider
if "api_key" not in st.session_state:
    st.session_state.api_key = None  # API key for cloud providers (None for Ollama)
if "llm_signature" not in st.session_state:
    st.session_state.llm_signature = (
        None  # (provider, model, api_key) the chain was built with
    )


# ── Helper: load embedding model once ────────────────────────────────────────


def get_cached_embedding_model():
    """Load the embedding model once per session and cache it in session state."""
    if st.session_state.embedding_model is None:
        with st.spinner("Loading embedding model (first run downloads ~90 MB)…"):
            st.session_state.embedding_model = get_embedding_model()
    return st.session_state.embedding_model


# ── Helper: provider/model selection -> LLM ──────────────────────────────────
# These wrap utils.llm_provider.build_llm() with whatever is currently selected
# in the sidebar. The retriever and vector store are never touched here —
# only the answering model is swapped, per the provider layer design.


def current_llm_signature():
    """Tuple identifying the currently selected provider configuration."""
    return (
        st.session_state.provider,
        st.session_state.model,
        st.session_state.api_key or "",
    )


def build_current_llm():
    """Build an LLM instance from the provider/model/API key chosen in the sidebar."""
    return build_llm(
        provider=st.session_state.provider,
        model=st.session_state.model,
        api_key=st.session_state.api_key,
    )


# ── Helper: process a PDF ────────────────────────────────────────────────────


def process_pdf(uploaded_file):
    """
    Full pipeline: PDF → documents → chunks → embeddings → FAISS → RAG chain.
    Updates session state and shows progress via Streamlit status widgets.
    """
    progress_bar = st.progress(0, text="Starting…")

    try:
        # Step 1: Load PDF
        progress_bar.progress(10, text="Reading PDF…")
        documents = load_pdf_from_upload(uploaded_file)
        meta = get_document_metadata(documents)

        # Step 2: Split into chunks
        progress_bar.progress(30, text="Splitting text into chunks…")
        chunks = split_documents(documents)
        chunk_stats = get_chunk_stats(chunks)

        # Step 3: Load embedding model
        progress_bar.progress(50, text="Loading embedding model…")
        embedding_model = get_cached_embedding_model()

        # Step 4: Build FAISS vector store
        progress_bar.progress(70, text="Building vector store…")
        vector_store = create_vector_store(chunks, embedding_model)

        # Step 5: Save vector store to disk
        progress_bar.progress(85, text="Saving index to disk…")
        save_vector_store(vector_store)

        # Step 6: Build the LLM for the selected provider, then the RAG chain
        progress_bar.progress(
            95, text=f"Initialising {st.session_state.provider} model…"
        )
        llm = build_current_llm()
        rag_chain = build_rag_chain(vector_store, llm=llm)
        st.session_state.llm_signature = current_llm_signature()

        # Save everything to session state
        st.session_state.vector_store = vector_store
        st.session_state.rag_chain = rag_chain
        st.session_state.pdf_processed = True
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.chat_history = []  # Reset chat on new PDF
        st.session_state.doc_stats = {**meta, **chunk_stats}

        progress_bar.progress(100, text="Done!")
        st.success(f"✅ **{uploaded_file.name}** processed successfully!")

    except Exception as e:
        progress_bar.empty()
        st.error(f"❌ Failed to process PDF: {str(e)}")
        raise


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 RAG Chatbot")
    st.markdown(
        f"*Retrieval-Augmented Generation · FAISS · {st.session_state.provider}*"
    )
    st.divider()

    # ---- Provider / Model Selection ----
    st.markdown("### ⚙️ Answer Model")

    provider = st.selectbox(
        "Provider",
        options=AVAILABLE_PROVIDERS,
        index=AVAILABLE_PROVIDERS.index(st.session_state.provider),
        help="Choose a local Ollama model or a cloud provider for answer generation.",
    )
    st.session_state.provider = provider

    if provider == "Ollama":
        current_model = (
            st.session_state.model
            if st.session_state.model in OLLAMA_MODELS
            else OLLAMA_MODELS[0]
        )
        model = st.selectbox(
            "Model",
            options=OLLAMA_MODELS,
            index=OLLAMA_MODELS.index(current_model),
        )
        st.session_state.model = model
        st.session_state.api_key = None
        st.caption("🖥️ Running locally via Ollama — no API key required.")
    else:
        st.session_state.model = DEFAULT_CLOUD_MODELS[provider]
        # Cache each provider's key separately so switching back and forth
        # within a session doesn't lose what was typed in.
        key_cache_field = f"_{provider}_key_cache"
        if key_cache_field not in st.session_state:
            st.session_state[key_cache_field] = ""

        api_key = st.text_input(
            f"{provider} API Key",
            type="password",
            value=st.session_state[key_cache_field],
            help="Used only for this session — never written to disk. "
            "You can also set this via an environment variable instead.",
        )
        st.session_state[key_cache_field] = api_key
        st.session_state.api_key = api_key

        st.caption(f"☁️ Using `{st.session_state.model}` via {provider}.")
        if not api_key:
            st.warning(f"Enter your {provider} API key to use this provider.")

    st.divider()

    # ---- PDF Upload ----
    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any PDF — research paper, manual, report, etc.",
    )

    if uploaded_file is not None:
        # Process only if it's a new file (avoids reprocessing on every rerun)
        if uploaded_file.name != st.session_state.pdf_name:
            if st.button("🔄 Process PDF", use_container_width=True, type="primary"):
                process_pdf(uploaded_file)

    st.divider()

    # ---- Load Existing Index ----
    st.markdown("### 💾 Saved Index")
    if vector_store_exists():
        if st.button("📂 Load saved index", use_container_width=True):
            try:
                with st.spinner("Loading saved vector store…"):
                    embedding_model = get_cached_embedding_model()
                    vs = load_vector_store(embedding_model=embedding_model)
                    llm = build_current_llm()
                    st.session_state.vector_store = vs
                    st.session_state.rag_chain = build_rag_chain(vs, llm=llm)
                    st.session_state.llm_signature = current_llm_signature()
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name = "Saved index"
                st.success("Index loaded!")
            except Exception as e:
                st.error(f"Could not load index: {e}")
    else:
        st.caption("No saved index found. Upload a PDF first.")

    st.divider()

    # ---- Document Stats ----
    if st.session_state.pdf_processed and st.session_state.doc_stats:
        st.markdown("### 📊 Document Stats")
        stats = st.session_state.doc_stats
        col1, col2 = st.columns(2)
        col1.metric("Pages", stats.get("total_pages", "—"))
        col2.metric("Chunks", stats.get("total_chunks", "—"))
        st.caption(f"Avg chunk: {stats.get('avg_chunk_size', '—')} chars")

    st.divider()

    # ---- Clear chat ----
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        # Rebuild chain to reset LangChain's internal memory
        if st.session_state.vector_store:
            try:
                llm = build_current_llm()
                st.session_state.rag_chain = build_rag_chain(
                    st.session_state.vector_store, llm=llm
                )
                st.session_state.llm_signature = current_llm_signature()
            except Exception as e:
                st.error(f"Could not rebuild chain: {e}")
        st.rerun()

    # ---- Status badge ----
    st.divider()
    if st.session_state.pdf_processed:
        st.markdown(
            f'<span class="badge-green">● Active: {st.session_state.pdf_name}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="badge-grey">○ No document loaded</span>',
            unsafe_allow_html=True,
        )


# ── Main Area ─────────────────────────────────────────────────────────────────

st.markdown("# 💬 RAG Chatbot")
st.markdown(
    "Ask any question about your uploaded PDF. "
    "Answers are grounded in the document's content."
)
st.divider()

# ---- Render chat history ----
chat_container = st.container()
with chat_container:
    if not st.session_state.chat_history:
        if st.session_state.pdf_processed:
            st.info("📝 Document loaded! Ask your first question below.")
        else:
            st.info("👈 Upload a PDF in the sidebar to get started.")
    else:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="user-bubble">🧑 {message["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="assistant-bubble">🤖 {message["content"]}</div>',
                    unsafe_allow_html=True,
                )
                # Show source documents if available
                if message.get("sources"):
                    with st.expander("📎 Source references", expanded=False):
                        for i, src in enumerate(message["sources"], 1):
                            page = src.metadata.get("page", "?")
                            snippet = src.page_content[:200].replace("\n", " ")
                            st.markdown(
                                f'<div class="source-box">'
                                f"<b>Ref {i}</b> · Page {page + 1}<br>{snippet}…"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

# ---- Chat input ----
st.divider()
user_question = st.chat_input(
    "Ask a question about your document…",
    disabled=not st.session_state.pdf_processed,
)

if user_question:
    if not st.session_state.rag_chain:
        st.error("No RAG chain available. Please process a PDF first.")
    else:
        # If the user switched provider/model/API key since the chain was
        # last built, rebuild it now — reusing the existing vector store
        # and retriever as-is, only swapping the answering model.
        chain_ready = True
        if current_llm_signature() != st.session_state.llm_signature:
            try:
                llm = build_current_llm()
                st.session_state.rag_chain = build_rag_chain(
                    st.session_state.vector_store, llm=llm
                )
                st.session_state.llm_signature = current_llm_signature()
            except Exception as e:
                st.error(f"⚠️ Could not switch to {st.session_state.provider}: {e}")
                chain_ready = False

        if chain_ready:
            # Append user message immediately
            st.session_state.chat_history.append(
                {"role": "user", "content": user_question}
            )

            # Get answer from RAG chain with a spinner
            with st.spinner("Searching document and generating answer…"):
                try:
                    result = query_rag_chain(st.session_state.rag_chain, user_question)
                    answer = result["answer"]
                    sources = result["source_documents"]
                except Exception as e:
                    answer = f"⚠️ Error generating answer: {str(e)}"
                    sources = []

            # Append assistant message
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )

            # Rerun to render the updated chat history
            st.rerun()
