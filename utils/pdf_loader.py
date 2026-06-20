"""
pdf_loader.py
-------------
Handles loading and extracting structured Markdown from PDF files.
Uses PyMuPDF4LLM instead of PyPDFLoader to preserve document structure:
  - Headings (H1, H2, H3 …)
  - Bullet and numbered lists
  - Tables (as GitHub-flavoured Markdown)
  - Sections and paragraph breaks

Supports both file path loading and in-memory Streamlit uploads.
Public API is unchanged from the original PyPDFLoader version:
  - load_pdf_from_path(file_path) -> list[Document]
  - load_pdf_from_upload(uploaded_file) -> list[Document]
  - get_document_metadata(documents) -> dict
"""

import tempfile
import os
from typing import TYPE_CHECKING  # Add type imports

if TYPE_CHECKING:
    pass  # type: ignore[import-untyped]
import pymupdf4llm  # type: ignore[import-untyped]
from langchain.schema import Document


# Add type ignore comments for third-party modules missing stubs
def load_pdf_from_path(file_path: str) -> list:
    """
    Load a PDF from a local file path and return a list containing one
    LangChain Document whose page_content is the full structured Markdown
    representation of the PDF.

    PyMuPDF4LLM extracts headings, tables, lists, and paragraphs as
    GitHub-flavoured Markdown — preserving the semantic structure that
    plain text extraction (PyPDFLoader) discards.

    Args:
        file_path (str): Absolute or relative path to the PDF file.

    Returns:
        list: List containing one LangChain Document with Markdown content
              and metadata (source, total_pages).

    Raises:
        FileNotFoundError: If the PDF file does not exist at the given path.
        ValueError: If no content can be extracted.
        Exception: For any other loading errors.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")

    # to_markdown() returns a single Markdown string for the entire document
    markdown_text = pymupdf4llm.to_markdown(file_path)  # type: ignore[import-untyped]

    if not markdown_text or not markdown_text.strip():
        raise ValueError("No content could be extracted from the PDF.")

    # Count pages via pymupdf for metadata
    import pymupdf  # type: ignore[import-untyped]

    with pymupdf.open(file_path) as pdf_doc:
        total_pages = pdf_doc.page_count  # type: ignore[import-untyped]

    doc = Document(
        page_content=markdown_text,
        metadata={
            "source": file_path,
            "total_pages": total_pages,
            # Keep 'page' key at 0 so source expander in app.py doesn't break
            "page": 0,
        },
    )
    return [doc]


def load_pdf_from_upload(uploaded_file) -> list:
    """
    Load a PDF from a Streamlit UploadedFile object.
    Writes the bytes to a temporary file, extracts structured Markdown via
    PyMuPDF4LLM, then cleans up the temp file.

    Args:
        uploaded_file: Streamlit UploadedFile object (from st.file_uploader).

    Returns:
        list: List containing one LangChain Document with Markdown content.

    Raises:
        ValueError: If no content is extracted from the uploaded PDF.
        Exception: For any other loading errors.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        markdown_text = pymupdf4llm.to_markdown(tmp_path)  # type: ignore[import-untyped]

        if not markdown_text or not markdown_text.strip():
            raise ValueError("No content could be extracted from the uploaded PDF.")

        import pymupdf  # type: ignore[import-untyped]

        with pymupdf.open(tmp_path) as pdf_doc:
            total_pages = pdf_doc.page_count  # type: ignore[import-untyped]

        doc = Document(
            page_content=markdown_text,
            metadata={
                "source": uploaded_file.name,
                "total_pages": total_pages,
                "page": 0,
            },
        )
        return [doc]

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def get_document_metadata(documents: list) -> dict:
    """
    Extract summary metadata from a list of loaded documents.
    Compatible with both old (per-page) and new (single-doc Markdown) formats.

    Args:
        documents (list): List of LangChain Document objects.

    Returns:
        dict: Metadata summary including page count and total characters.
    """
    total_chars = sum(len(doc.page_content) for doc in documents)
    # Try to get total_pages from metadata (set by new loader); fall back to len
    total_pages = (
        documents[0].metadata.get("total_pages", len(documents)) if documents else 0
    )
    return {
        "total_pages": total_pages,
        "total_characters": total_chars,
        "average_chars_per_page": total_chars // total_pages if total_pages else 0,
    }
