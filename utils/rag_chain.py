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
  - build_rag_chain(vector_store, llm=None) -> ConversationalRetrievalChain
  - query_rag_chain(chain, question) -> dict
"""

from langchain_ollama import ChatOllama
from langchain.chains.conversational_retrieval.base import (
    ConversationalRetrievalChain,
)
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


# ----- Prompt Template --------------------------------------------------------

RAG_PROMPT_TEMPLATE = """You are a helpful assistant
that answers questions based strictly
on the provided document context. If the answer is not in the context, say:
"I couldn't find that information in the uploaded document."

Context from document:
{context}

Chat History:
{chat_history}

Question: {question}

Provide a clear, concise answer based only on the context above."""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=RAG_PROMPT_TEMPLATE,
)


# ----- LLM Setup --------------------------------------------------------------


def get_llm(
    model_name: str = "qwen3:4b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.3,
):
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
    llm = ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
    )
    return llm


# ----- RAG Chain --------------------------------------------------------------


def build_rag_chain(vector_store, llm=None) -> ConversationalRetrievalChain:
    """
    Build a ConversationalRetrievalChain backed by a local Ollama model.

    Uses FAISS retriever to fetch relevant Markdown chunks, then passes those
    chunks along with conversation history to qwen3:4b for answer generation.

    Args:
        vector_store (FAISS): The loaded FAISS vector store.
        llm: An optional pre-built LLM instance. If None, Ollama qwen3:4b
             is initialised automatically.

    Returns:
        ConversationalRetrievalChain: A ready-to-use RAG chain.
    """
    if llm is None:
        llm = get_llm()

    # Retriever: returns the top 4 most similar chunks for each query
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    # Memory: keeps the conversation history in context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # Chain: ties retriever + memory + LLM together
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": RAG_PROMPT},
        verbose=False,
    )

    return chain


def query_rag_chain(chain: ConversationalRetrievalChain, question: str) -> dict:
    """
    Run a question through the RAG chain and return the answer + sources.

    Args:
        chain (ConversationalRetrievalChain): The built RAG chain.
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

    result = chain.invoke({"question": question})
    return {
        "answer": result.get("answer", "No answer returned."),
        "source_documents": result.get("source_documents", []),
    }
