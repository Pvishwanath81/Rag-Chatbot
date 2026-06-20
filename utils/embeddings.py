"""
embeddings.py
-------------
Creates a HuggingFace sentence-transformer embedding model.
The embedding model converts text chunks into dense vector representations
that can be stored and searched in FAISS.

Model used: all-MiniLM-L6-v2
  - Lightweight (22M parameters), fast, and high-quality for semantic search.
  - Produces 384-dimensional embeddings.
  - No API key required — runs locally.
"""

from langchain_huggingface import HuggingFaceEmbeddings


# Default model — good balance of speed and quality for RAG
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> HuggingFaceEmbeddings:
    """
    Initialize and return a HuggingFace embedding model.

    The model is downloaded on first use and cached locally by HuggingFace.
    Subsequent calls load from cache without re-downloading.

    Args:
        model_name (str): HuggingFace model identifier.
                          Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.

    Returns:
        HuggingFaceEmbeddings: A LangChain-compatible embedding object.

    Raises:
        Exception: If the model cannot be downloaded or initialized.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # Use "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},  # Normalise for cosine similarity
    )
    return embedding_model
