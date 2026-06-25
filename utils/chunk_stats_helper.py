"""Helper utilities for chunk statistics calculations."""

from typing import List, Dict
from langchain_core.documents import Document


def get_base_chunk_statistics(chunks: List[Document]) -> Dict:
    """
    Calculates basic statistics about a list of document chunks.

    Args:
        chunks: List of chunked LangChain Documents.

    Returns:
        dict: Stats including total chunk count, min/max/avg chunk size.
    """
    if not chunks:
        return {"total_chunks": 0}

    sizes = [len(chunk.page_content) for chunk in chunks]
    return {
        "total_chunks": len(chunks),
        "min_chunk_size": min(sizes),
        "max_chunk_size": max(sizes),
        "avg_chunk_size": sum(sizes) // len(sizes),
    }
