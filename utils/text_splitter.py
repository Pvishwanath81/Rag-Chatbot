"""
text_splitter.py
----------------
Splits Markdown-formatted documents into semantically coherent chunks.

Why MarkdownTextSplitter instead of RecursiveCharacterTextSplitter?
--------------------------------------------------------------------
PyMuPDF4LLM outputs a single structured Markdown string for the entire PDF,
preserving headings, tables, and lists. A generic character-based splitter
is blind to this structure and can:
  - Cut a table in the middle of a row
  - Separate a heading from its first paragraph
  - Split a bullet list mid-item

MarkdownTextSplitter treats Markdown headings (#, ##, ###) as primary split
points, keeping each section intact before falling back to size-based splits.
This results in chunks that carry their heading context and remain semantically
self-contained, improving retrieval accuracy.

Public API is unchanged from the original version:
  - split_documents(documents, chunk_size, chunk_overlap) -> list[Document]
  - get_chunk_stats(chunks) -> dict
"""

from typing import Any, cast
from langchain_text_splitters import MarkdownTextSplitter


def split_documents(
    documents: list, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list:
    """
    Split a list of Markdown LangChain Document objects into smaller chunks.

    MarkdownTextSplitter splits at Markdown heading boundaries first
    (h1 > h2 > h3 > paragraph > line > word > character), keeping sections
    together before falling back to size limits.

    Args:
        documents (list): List of LangChain Document objects (from pdf_loader).
                          Expected to contain Markdown-formatted page_content.
        chunk_size (int): Maximum number of characters per chunk. Default 1000.
        chunk_overlap (int): Overlapping characters between consecutive chunks.
                             Preserves context at chunk boundaries. Default 200.

    Returns:
        list: List of smaller LangChain Document chunks, each inheriting the
              source document's metadata.

    Raises:
        ValueError: If the documents list is empty or splitting yields nothing.
    """
    if not documents:
        raise ValueError("Cannot split an empty document list.")

    splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = cast(list[Any], splitter.split_documents(documents))

    if not chunks:
        raise ValueError(
            "Text splitting produced no chunks. "
            "The PDF may be empty or contain only images."
        )

    return chunks


def get_chunk_stats(chunks: list) -> dict:
    """
    Return statistics about the produced chunks for display in the UI.

    Args:
        chunks (list): List of chunked LangChain Documents.

    Returns:
        dict: Stats including chunk count, min/max/avg chunk size in characters.
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
