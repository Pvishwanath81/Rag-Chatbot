"""
vector_store.py
---------------
Handles creation, saving, and loading of a FAISS vector store.

FAISS (Facebook AI Similarity Search) enables fast nearest-neighbour lookup
over dense vectors. Here we use LangChain's FAISS wrapper which manages both
the raw FAISS index and the document metadata together.
"""

import os
from langchain_community.vectorstores import FAISS
from utils.embeddings import get_embedding_model


# Default directory where the FAISS index is persisted on disk
DEFAULT_VECTORSTORE_PATH = "vectorstore/faiss_index"


def create_vector_store(chunks: list, embedding_model=None) -> FAISS:
    """
    Build a new FAISS vector store from a list of document chunks.

    Each chunk's text is embedded using the provided (or default) embedding model,
    and the resulting vectors are stored in a FAISS flat index.

    Args:
        chunks (list): List of LangChain Document objects (chunked text).
        embedding_model: A LangChain-compatible embedding object.
                         If None, the default HuggingFace model is used.

    Returns:
        FAISS: An in-memory FAISS vector store ready for similarity search.

    Raises:
        ValueError: If chunks is empty.
        Exception: If embedding or indexing fails.
    """
    if not chunks:
        raise ValueError("Cannot build a vector store from an empty chunk list.")

    if embedding_model is None:
        embedding_model = get_embedding_model()

    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store


def save_vector_store(
    vector_store: FAISS, path: str = DEFAULT_VECTORSTORE_PATH
) -> None:
    """
    Persist a FAISS vector store to disk so it can be reloaded later.

    Saves two files under `path/`:
      - index.faiss  — the raw FAISS binary index
      - index.pkl    — the docstore and index-to-docstore ID mapping

    Args:
        vector_store (FAISS): The in-memory FAISS vector store to save.
        path (str): Directory path where the index files will be written.

    Raises:
        Exception: If the directory cannot be created or files cannot be written.
    """
    os.makedirs(path, exist_ok=True)
    vector_store.save_local(path)


def load_vector_store(
    path: str = DEFAULT_VECTORSTORE_PATH, embedding_model=None
) -> FAISS:
    """
    Load a previously saved FAISS vector store from disk.

    Args:
        path (str): Directory path where the index files were saved.
        embedding_model: The same embedding model used when the index was built.
                         Must produce vectors of the same dimensionality.

    Returns:
        FAISS: The restored FAISS vector store.

    Raises:
        FileNotFoundError: If no saved index exists at the given path.
        Exception: If loading fails.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved vector store found at: {path}")

    if embedding_model is None:
        embedding_model = get_embedding_model()

    # allow_dangerous_deserialization=True is required by newer LangChain versions
    # because pickle is used internally — safe here since we control the file.
    vector_store = FAISS.load_local(
        path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    return vector_store


def vector_store_exists(path: str = DEFAULT_VECTORSTORE_PATH) -> bool:
    """
    Check whether a saved FAISS index already exists at the given path.

    Args:
        path (str): Directory path to check.

    Returns:
        bool: True if both index files exist, False otherwise.
    """
    return os.path.exists(os.path.join(path, "index.faiss")) and os.path.exists(
        os.path.join(path, "index.pkl")
    )
