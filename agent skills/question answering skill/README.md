# Question Answering Skill

## Overview

The **Question Answering (QA) Skill** is the generative core of the RAG chatbot pipeline. It takes a user's natural language question alongside retrieved document context and produces a precise, grounded, and fluent answer using a large language model. This skill integrates prompt engineering, context injection, and response validation to ensure answers are factually consistent with the source documents and appropriately scoped to the available knowledge.

---

## Purpose

The QA Skill is where retrieval meets generation — the defining capability of any RAG system. Without a robust QA layer, even perfectly retrieved documents cannot be transformed into actionable answers. This skill ensures the LLM does not hallucinate by anchoring responses strictly to retrieved context, enforces response formatting, handles unanswerable queries gracefully, and maintains conversation history for multi-turn dialogue. It is the primary interface between the knowledge base and the end user.

---

## Inputs

| Input | Type | Description |
|-------|------|-------------|
| `question` | `str` | The user's natural language question |
| `retrieved_context` | `List[str]` | Relevant document chunks from the Document Search Skill |
| `chat_history` | `List[dict]` | Previous turns in the conversation for multi-turn QA (optional) |
| `llm` | `LangChain LLM object` | The language model used for answer generation |
| `prompt_template` | `str` | Custom instruction prompt to guide answer style and constraints |
| `max_answer_tokens` | `int` | Maximum length of the generated answer (default: 512) |

---

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `answer` | `str` | The generated, context-grounded answer to the user's question |
| `confidence_level` | `str` | Qualitative confidence: `"high"`, `"medium"`, or `"low"` |
| `source_references` | `List[str]` | List of source documents used to generate the answer |
| `is_answerable` | `bool` | Whether the question could be answered from the provided context |

---

## Workflow

1. **Context Preparation** — Retrieved document chunks are cleaned, deduplicated, and formatted into a numbered context block.
2. **Prompt Construction** — The user's question, context block, and conversation history are injected into a structured prompt template using LangChain's `PromptTemplate` or `ChatPromptTemplate`.
3. **Answerability Check** — A pre-generation heuristic (or LLM-based check) evaluates whether the context contains sufficient information to answer the question.
4. **LLM Inference** — The assembled prompt is sent to the LLM (e.g., Gemini 1.5 Flash, GPT-4o-mini) for answer generation.
5. **Answer Extraction** — The raw LLM response is parsed to isolate the answer text, stripping any reasoning traces or verbose preambles.
6. **Confidence Assignment** — Based on the similarity scores of retrieved chunks and the LLM's hedging language, a confidence level is assigned.
7. **Source Attribution** — Source document names from the retrieved context metadata are appended to the response for transparency.
8. **History Update** — The current Q&A pair is appended to `chat_history` for use in subsequent turns.

---

## Example Input

```python
question = "What machine learning algorithm is used for anomaly detection in network traffic?"

retrieved_context = [
    "Isolation Forest is an unsupervised algorithm effective for anomaly detection in high-dimensional data...",
    "In network security, models like Autoencoder-based detectors and One-Class SVM are commonly used...",
    "LSTM networks have shown strong performance in detecting time-series anomalies in network packet flows..."
]

chat_history = [
    {"role": "user", "content": "What is anomaly detection?"},
    {"role": "assistant", "content": "Anomaly detection identifies unusual patterns that deviate from expected behavior..."}
]
```

---

## Example Output

```json
{
  "answer": "For anomaly detection in network traffic, several machine learning algorithms are commonly used. Isolation Forest is widely applied due to its efficiency with high-dimensional data. Autoencoder-based neural networks and One-Class SVM are also used in network security contexts. For time-series network data, LSTM networks have demonstrated strong detection capabilities by learning sequential packet flow patterns.",
  "confidence_level": "high",
  "source_references": [
    "network_security_handbook.pdf — Page 43",
    "ml_for_cybersecurity.pdf — Page 117"
  ],
  "is_answerable": true
}
```

---

## Use Cases

1. **IT Helpdesk Automation** — Answer employee IT queries from internal runbooks and troubleshooting guides without human agent intervention.
2. **Educational Tutoring** — Answer student questions grounded in course textbooks and lecture notes uploaded to the knowledge base.
3. **Healthcare Q&A** — Respond to clinical or patient queries using verified medical literature and hospital protocols.
4. **Financial Advisory Chatbot** — Answer investor questions about fund performance, regulations, or market conditions from proprietary reports.
5. **Government Services** — Enable citizens to query tax codes, welfare eligibility rules, and public service information in plain language.
6. **Competitive Intelligence** — Answer internal strategy questions grounded in market research, analyst reports, and competitor filings.

---

## Benefits

- **Hallucination Mitigation** — Restricting the LLM to retrieved context dramatically reduces factually incorrect responses.
- **Multi-Turn Awareness** — Conversation history enables coherent follow-up questions without losing context.
- **Graceful Degradation** — When context is insufficient, the skill returns a transparent "cannot answer" response rather than fabricating one.
- **Prompt Customizability** — Domain-specific prompt templates allow the skill to be tuned for tone, format, and constraints per use case.
- **Source Transparency** — Answer attribution builds user trust by pointing to verifiable source documents.

---

## Future Enhancements

- **Chain-of-Thought Prompting** — Instruct the LLM to reason step-by-step before answering for improved accuracy on complex queries.
- **Answer Verification Layer** — Add a secondary LLM call to fact-check the generated answer against retrieved context.
- **Multilingual QA** — Extend the skill to answer questions in multiple languages by incorporating multilingual embedding models.
- **Structured Answer Formats** — Support output schemas like JSON, tables, or numbered steps for integration with downstream UIs.
- **Confidence Calibration** — Train a lightweight classifier on LLM logits to produce calibrated numerical confidence scores.

---

## Dependencies

| Library / Tool | Purpose |
|----------------|---------|
| `langchain` | QA chains, prompt templates, memory management |
| `google-generativeai` | Gemini 1.5 Flash as the primary LLM backend |
| `openai` | Optional GPT-4o / GPT-4o-mini integration |
| `streamlit` | Frontend interface for the chatbot |
| `langchain-community` | Document loaders and retriever wrappers |
| `python-dotenv` | Secure API key management via `.env` files |

---

> **Maintainer:** Team Trident | GITAM University, Hyderabad
> **Project:** RAG Chatbot — Algonive Internship 2024–25
> **License:** MIT
