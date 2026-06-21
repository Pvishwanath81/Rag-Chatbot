"""
llm_provider.py
----------------
Provider factory for the answer-generation LLM.

This module is the ONLY place that knows how to instantiate a chat model.
It deliberately knows nothing about retrieval, chunking, embeddings, or
FAISS — those stay exactly as they were. `build_rag_chain()` in
rag_chain.py already accepts an externally-built `llm`, so this module
simply produces one and the rest of the app is unaffected.

Supported providers:
    - Ollama   -> langchain_ollama.ChatOllama        (local, no API key)
    - Gemini   -> langchain_google_genai.ChatGoogleGenerativeAI
    - OpenAI   -> langchain_openai.ChatOpenAI
    - Claude   -> langchain_anthropic.ChatAnthropic
    - Groq     -> langchain_groq.ChatGroq

Public API:
    build_llm(provider, model=None,
    api_key=None, temperature=0.3,
    base_url=...) -> BaseChatModel
    AVAILABLE_PROVIDERS      -> list[str]
    OLLAMA_MODELS            -> list[str]
    DEFAULT_CLOUD_MODELS     -> dict[str, str]
    PROVIDER_ENV_VARS        -> dict[str, str]
"""

import os
from pydantic import SecretStr

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

# ----- Provider / model catalogue ---------------------------------------------

AVAILABLE_PROVIDERS = ["Ollama", "Gemini", "OpenAI", "Claude", "Groq"]

# Models offered for the local Ollama provider (must already be pulled).
OLLAMA_MODELS = [
    "qwen3:4b",
    "qwen2.5-coder:7b",
    "qwen2.5:7b",
    "llama3",
    "mistral",
]

# One sensible default model per cloud provider. The UI only asks for an
# API key for cloud providers (per spec), so the model is chosen here.
# Update these if a provider retires a model.
DEFAULT_CLOUD_MODELS = {
    "Gemini": "gemini-2.5-flash",
    "OpenAI": "gpt-4o-mini",
    "Claude": "claude-sonnet-4-6",
    "Groq": "llama-3.3-70b-versatile",
}

# Environment variable each provider's key can fall back to if the user
# doesn't type one into the sidebar (handy for local dev / .env files).
PROVIDER_ENV_VARS = {
    "Gemini": "GOOGLE_API_KEY",
    "OpenAI": "OPENAI_API_KEY",
    "Claude": "ANTHROPIC_API_KEY",
    "Groq": "GROQ_API_KEY",
}

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def build_llm(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.3,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
):
    """
    Build and return a LangChain chat model for the requested provider.

    The retriever / vector store are never touched here — this function's
    only job is to produce something with a `.invoke()` chat interface
    that `build_rag_chain()` can plug in as the answering model.

    Args:
        provider (str): One of AVAILABLE_PROVIDERS ("Ollama", "Gemini",
                         "OpenAI", "Claude", "Groq"). Case-insensitive.
        model (str): Model name/tag. If None, a sensible default is used
                     (OLLAMA_MODELS[0] for Ollama, DEFAULT_CLOUD_MODELS[...]
                     for cloud providers).
        api_key (str): API key for cloud providers. If None/empty, falls
                        back to the provider's environment variable (see
                        PROVIDER_ENV_VARS). Not required for Ollama.
        temperature (float): Sampling temperature. Default 0.3.
        base_url (str): Ollama server URL. Only used when provider == "Ollama".

    Returns:
        A LangChain BaseChatModel instance (ChatOllama / ChatGoogleGenerativeAI /
        ChatOpenAI / ChatAnthropic / ChatGroq).

    Raises:
        ValueError: If the provider is unknown, or a cloud provider is
                    selected without an API key available (sidebar input
                    or environment variable).
        Exception: If the underlying provider SDK fails to initialise
                   (e.g. missing package, invalid key format).
    """
    if not provider:
        raise ValueError("No provider specified.")

    provider_key = provider.strip().lower()
    matches = [p for p in AVAILABLE_PROVIDERS if p.lower() == provider_key]
    if not matches:
        raise ValueError(
            f"Unknown provider: '{provider}'. Expected one of {AVAILABLE_PROVIDERS}."
        )
    canonical_name = matches[0]

    # ----- Local: Ollama (no API key) -----
    if provider_key == "ollama":
        return ChatOllama(
            model=model or OLLAMA_MODELS[0],
            base_url=base_url,
            temperature=temperature,
        )

    # ----- Cloud providers: resolve the API key (sidebar input > env var) -----
    resolved_key = api_key or os.getenv(PROVIDER_ENV_VARS.get(canonical_name, ""), "")

    if not resolved_key:
        raise ValueError(
            f"{canonical_name} requires an API key. Enter one in the sidebar "
            f"or set the {PROVIDER_ENV_VARS.get(canonical_name)} environment variable."
        )

    if provider_key == "gemini":
        return ChatGoogleGenerativeAI(
            model=model or DEFAULT_CLOUD_MODELS["Gemini"],
            google_api_key=SecretStr(resolved_key),
            temperature=temperature,
        )

    if provider_key == "openai":
        return ChatOpenAI(
            model=model or DEFAULT_CLOUD_MODELS["OpenAI"],
            api_key=SecretStr(resolved_key),
            temperature=temperature,
        )

    if provider_key == "claude":
        # ChatAnthropic uses 'anthropic_model_name' parameter, not 'model'
        return ChatAnthropic(
            model_name=model or DEFAULT_CLOUD_MODELS["Claude"],
            api_key=SecretStr(resolved_key),
            temperature=temperature,
            timeout=None,
            stop=None,
        )

    if provider_key == "groq":
        return ChatGroq(
            model=model or DEFAULT_CLOUD_MODELS["Groq"],
            api_key=SecretStr(resolved_key),
            temperature=temperature,
        )

    raise ValueError(f"Unsupported provider after validation: {provider}")
