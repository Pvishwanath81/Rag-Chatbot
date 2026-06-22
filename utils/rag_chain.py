"""
rag_chain.py
------------
Builds the RAG (Retrieval-Augmented Generation) chain using LangChain + Ollama.

Gemini has been removed. The LLM is now served locally via Ollama (qwen3:4b).
No API keys, no network calls to cloud services.

The chain works as follows:
  1. User asks a question.
  2. FAISS retriever fetches the top-k most relevant document chunks.
  3. Those chunks are injected into a prompt as "context."
  4. Ollama (qwen3:4b) generates an answer grounded in that context.
  5. Source documents used for the answer are returned for transparency.

Public API is unchanged:
  - build_rag_chain(vector_store, llm=None) -> dict (chain state)
  - query_rag_chain(chain, question) -> dict
"""

from operator import itemgetter
from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama


# ----- Prompt Template --------------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant that answers questions based strictly
on the provided document context. If the answer is not in the context, say:
"I couldn't find that information in the uploaded document."

Context from document:
{context}""",
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
        temperature (float): Controls randomness. Lower = more factual. Default 0.3.

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


def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return (or create) an in-memory chat history for a session."""
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]


# ----- RAG Chain --------------------------------------------------------------


def build_rag_chain(vector_store: Any, llm: Any = None) -> dict:
    """
    Build a RAG chain backed by a local Ollama model using LCEL.

    Uses FAISS retriever to fetch relevant chunks, then passes those
    chunks along with conversation history to qwen3:4b for answer generation.

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
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL chain: retrieve → format → prompt → llm → parse
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
