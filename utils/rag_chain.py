"""
rag_chain.py
------------
Enhanced RAG (Retrieval-Augmented Generation) chain using LangChain + Ollama.
Now supports multiple document formats: PDF, DOCX, and TXT files.

The chain works as follows:
  1. User uploads multiple documents (PDF, DOCX, TXT).
  2. Documents are processed and combined into a unified vector store.
  3. User asks a question.
  4. FAISS retriever fetches the top-k most relevant document chunks.
  5. Those chunks are injected into a prompt as "context."
  6. Ollama (qwen3:4b) generates an answer grounded in that context.
  7. Source documents used for the answer are returned for transparency.

Public API:
  - load_documents_from_uploads(uploaded_files) -> list[Document]
  - build_rag_chain(vector_store, llm=None) -> dict (chain state)
  - query_rag_chain(chain, question) -> dict
"""

import os
from operator import itemgetter
from typing import Any, List, Dict

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama

import pymupdf
import pymupdf4llm
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)

from utils.chunk_stats_helper import get_base_chunk_statistics


# ----- Document Loading Functions ---------------------------------------------


def load_pdf_from_file(filename: str, filepath: str) -> List[Document]:
    """
    Load a PDF from a file path on disk.

    Args:
        filename: Original name of the file (for metadata).
        filepath: Path to the PDF file on disk.

    Returns:
        list: List containing one LangChain Document with Markdown content.

    Raises:
        ValueError: If no content is extracted from the uploaded PDF.
        RuntimeError: For any other loading errors.
    """
    try:
        markdown_text = pymupdf4llm.to_markdown(filepath)
        if not markdown_text or not markdown_text.strip():
            raise ValueError(f"No content could be extracted from the PDF: {filename}")

        with pymupdf.open(filepath) as pdf_doc:
            total_pages = pdf_doc.page_count
            doc = Document(
                page_content=markdown_text,
                metadata={
                    "source": filename,
                    "file_type": "pdf",
                    "total_pages": total_pages,
                    "page": 0,
                },
            )
        return [doc]
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error processing PDF {filename}: {e}") from e


def load_docx_from_file(filename: str, filepath: str) -> List[Document]:
    """
    Load a DOCX file from a file path on disk.

    Args:
        filename: Original name of the file (for metadata).
        filepath: Path to the DOCX file on disk.

    Returns:
        list: List containing one LangChain Document with text content.

    Raises:
        ValueError: If no content is extracted from the uploaded DOCX.
        RuntimeError: For any other loading errors.
    """
    try:
        loader = Docx2txtLoader(filepath)
        documents = loader.load()

        if not documents or not documents[0].page_content.strip():
            raise ValueError(f"No content could be extracted from the DOCX: {filename}")

        # Update metadata
        for doc in documents:
            doc.metadata["source"] = filename
            doc.metadata["file_type"] = "docx"
            doc.metadata["total_pages"] = 1

        return list(documents)
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error processing DOCX {filename}: {e}") from e


def load_txt_from_file(filename: str, filepath: str) -> List[Document]:
    """
    Load a TXT file from a file path on disk.

    Args:
        filename: Original name of the file (for metadata).
        filepath: Path to the TXT file on disk.

    Returns:
        list: List containing one LangChain Document with text content.

    Raises:
        ValueError: If no content is extracted from the uploaded TXT.
        RuntimeError: For any other loading errors.
    """
    try:
        with open(filepath, "rb") as f:
            content = f.read()

        # Try to decode as UTF-8, fallback to other encodings if needed
        try:
            text_content = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text_content = content.decode("latin-1")
            except UnicodeDecodeError:
                text_content = content.decode("utf-8", errors="ignore")

        if not text_content.strip():
            raise ValueError(f"No content could be extracted from the TXT: {filename}")

        doc = Document(
            page_content=text_content,
            metadata={
                "source": filename,
                "file_type": "txt",
                "total_pages": 1,
                "character_count": len(text_content),
            },
        )

        return [doc]
    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"Error loading TXT file {filename}: {e}") from e


def load_documents_from_uploads(uploaded_files) -> List[Document]:
    """
    Load multiple documents from uploaded file objects.
    Supports PDF, DOCX, and TXT formats.

    Args:
        uploaded_files: List of (filename, filepath) tuples where filename is
                         the original file name and filepath is the path to a
                         temp file on disk containing the file bytes.

    Returns:
        list: List of LangChain Document objects from all uploaded files.

    Raises:
        ValueError: If no files are provided or no content is extracted.
        RuntimeError: For any loading errors.
    """
    if not uploaded_files:
        raise ValueError("No files provided for loading.")

    all_documents: List[Document] = []
    supported_extensions = {".pdf", ".docx", ".txt"}

    for filename, filepath in uploaded_files:
        file_extension = os.path.splitext(filename)[1].lower()

        if file_extension not in supported_extensions:
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {', '.join(sorted(supported_extensions))}"
            )

        documents: List[Document] = []
        try:
            if file_extension == ".pdf":
                documents = load_pdf_from_file(filename, filepath)
            elif file_extension == ".docx":
                documents = load_docx_from_file(filename, filepath)
            elif file_extension == ".txt":
                documents = load_txt_from_file(filename, filepath)

            all_documents.extend(documents)

        except Exception as e:
            raise RuntimeError(f"Error processing {filename}: {e}") from e

    if not all_documents:
        raise ValueError(
            "No content could be extracted from any of the uploaded files."
        )

    return all_documents


def split_documents_by_type(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split documents into chunks using appropriate splitters based on content type.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Overlapping characters between consecutive chunks.

    Returns:
        list: List of chunked LangChain Document objects.

    Raises:
        ValueError: If the documents list is empty or splitting yields nothing.
    """
    if not documents:
        raise ValueError("Cannot split an empty document list.")

    all_chunks: List[Document] = []

    # Separate documents by type for appropriate splitting
    pdf_docs = [doc for doc in documents if doc.metadata.get("file_type") == "pdf"]
    other_docs = [doc for doc in documents if doc.metadata.get("file_type") != "pdf"]

    # Use MarkdownTextSplitter for PDF documents (which contain Markdown)
    if pdf_docs:
        markdown_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        pdf_chunks = markdown_splitter.split_documents(pdf_docs)
        all_chunks.extend(pdf_chunks)

    # Use RecursiveCharacterTextSplitter for DOCX and TXT documents
    if other_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        other_chunks = text_splitter.split_documents(other_docs)
        all_chunks.extend(other_chunks)

    if not all_chunks:
        raise ValueError(
            "Text splitting produced no chunks. "
            "The documents may be empty or contain only images."
        )

    return all_chunks


def get_documents_metadata(documents: List[Document]) -> Dict:
    """
    Extract summary metadata from a list of loaded documents.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        dict: Metadata summary including file counts, page counts,
              and character counts.
    """
    if not documents:
        return {
            "total_files": 0,
            "total_pages": 0,
            "total_characters": 0,
            "file_types": {},
        }

    total_chars = sum(len(doc.page_content) for doc in documents)
    total_pages = sum(doc.metadata.get("total_pages", 1) for doc in documents)

    # Count files by type
    file_types: dict[str, int] = {}
    file_names = set()

    for doc in documents:
        file_type = doc.metadata.get("file_type", "unknown")
        file_name = doc.metadata.get("source", "unknown")

        file_names.add(file_name)
        file_types[file_type] = file_types.get(file_type, 0) + 1

    return {
        "total_files": len(file_names),
        "total_pages": total_pages,
        "total_characters": total_chars,
        "average_chars_per_page": total_chars // total_pages if total_pages else 0,
        "file_types": file_types,
    }


# ----- Prompt Template --------------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a helpful assistant that answers questions based strictly "
                "on the provided document context from multiple uploaded files. "
                "If the answer is not in the context, say: "
                '"I couldn\'t find that information in the uploaded documents." '
                "When referencing information, mention the source document name "
                "when possible.\n\nContext from documents:\n{context}"
            ),
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


# ----- LLM Setup --------------------------------------------------------------


def get_llm(
    model_name: str = "qwen3:4b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.3,
) -> ChatOllama:
    """
    Initialise and return a local Ollama LLM via LangChain.

    Ollama must be running locally (ollama serve) with the model pulled:
        ollama pull qwen3:4b

    Args:
        model_name (str): Ollama model tag. Default 'qwen3:4b'.
        base_url (str): Ollama server URL. Default 'http://localhost:11434'.
        temperature (float): Controls randomness. Lower = more factual.
                             Default 0.3.

    Returns:
        ChatOllama: LangChain-wrapped local Ollama chat model.

    Raises:
        Exception: If Ollama is not running or the model is not pulled.
    """
    return ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
    )


# ----- Session store for conversation memory ----------------------------------

_session_store: dict[str, InMemoryChatMessageHistory] = {}


def _get_session_history(
    session_id: str,
) -> InMemoryChatMessageHistory:
    """Return (or create) an in-memory chat history for a session."""
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]


# ----- RAG Chain --------------------------------------------------------------


def build_rag_chain(vector_store: Any, llm: Any = None) -> dict:
    """
    Build a RAG chain backed by a local Ollama model using LCEL.

    Uses FAISS retriever to fetch relevant chunks from multiple document
    types, then passes those chunks along with conversation history to
    qwen3:4b for answer generation.

    Args:
        vector_store (FAISS): The loaded FAISS vector store.
        llm: An optional pre-built LLM instance. If None, Ollama qwen3:4b
             is initialised automatically.

    Returns:
        dict with keys:
            - "chain": RunnableWithMessageHistory ready to invoke.
            - "retriever": The FAISS retriever (for source doc access).
    """
    if llm is None:
        llm = get_llm()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    def format_docs(docs: list) -> str:
        """Format retrieved documents with source information."""
        formatted_chunks = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            file_type = doc.metadata.get("file_type", "unknown")
            content = doc.page_content

            formatted_chunks.append(
                f"[Source: {source} ({file_type.upper()})]\n{content}"
            )

        return "\n\n---\n\n".join(formatted_chunks)

    # LCEL chain: retrieve -> format -> prompt -> llm -> parse
    rag_core = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | RunnableLambda(format_docs)
        )
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    chain_with_history = RunnableWithMessageHistory(
        rag_core,
        _get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return {"chain": chain_with_history, "retriever": retriever}


def query_rag_chain(chain: dict, question: str) -> dict:
    """
    Run a question through the RAG chain and return the answer + sources.

    Args:
        chain (dict): The built RAG chain dict from build_rag_chain().
        question (str): The user's question string.

    Returns:
        dict with keys:
            - "answer" (str): Ollama's generated answer.
            - "source_documents" (list): LangChain Documents used as context.

    Raises:
        ValueError: If question is empty.
        Exception: If the chain invocation fails (e.g., Ollama not running).
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    runnable: RunnableWithMessageHistory = chain["chain"]
    retriever = chain["retriever"]

    config = {"configurable": {"session_id": "default"}}

    answer = runnable.invoke({"question": question}, config=config)
    source_documents = retriever.invoke(question)

    return {
        "answer": answer,
        "source_documents": source_documents,
    }


def get_chunk_stats(chunks: List[Document]) -> Dict:
    """
    Return statistics about the produced chunks for display in the UI.
    Enhanced to show file type distribution.

    Args:
        chunks: List of chunked LangChain Documents.

    Returns:
        dict: Stats including chunk count, min/max/avg chunk size,
              and file type distribution.
    """
    if not chunks:
        return {"total_chunks": 0}

    base_stats = get_base_chunk_statistics(chunks)

    # Count chunks by file type
    file_type_counts: dict[str, int] = {}
    for chunk in chunks:
        file_type = chunk.metadata.get("file_type", "unknown")
        file_type_counts[file_type] = file_type_counts.get(file_type, 0) + 1

    base_stats["file_type_distribution"] = file_type_counts
    return base_stats
