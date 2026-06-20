# Document Summarization Skill

## Overview

The **Document Summarization Skill** condenses retrieved document chunks or full documents into concise, coherent summaries using a large language model. Operating as a post-retrieval processing layer in the RAG pipeline, this skill reduces cognitive load on both the end user and the downstream generation model by distilling lengthy content into structured, digestible summaries. It supports both extractive (key sentence selection) and abstractive (LLM-rewritten) summarization modes.

---

## Purpose

In a RAG chatbot, retrieved chunks may be lengthy, redundant, or contain information beyond the scope of the user's query. The Document Summarization Skill ensures that only the **most relevant and compressed information** is presented to the user or passed as context to the answer generation stage. This leads to cleaner responses, reduced token usage, and improved user experience. It also enables standalone use cases like document briefings, executive summaries, and multi-document synthesis.

---

## Inputs

| Input | Type | Description |
|-------|------|-------------|
| `document_chunks` | `List[str]` | Retrieved text chunks from the Document Search Skill |
| `query` | `str` | Original user query to guide query-focused summarization (optional) |
| `summary_mode` | `str` | `"abstractive"` (default) or `"extractive"` |
| `max_summary_length` | `int` | Maximum word/token count of the output summary |
| `llm` | `LangChain LLM object` | The language model used for abstractive summarization |

---

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `summary` | `str` | Condensed summary of the input document or chunks |
| `key_points` | `List[str]` | Bullet-point highlights extracted from the content |
| `token_reduction_ratio` | `float` | Ratio of output tokens to input tokens (compression metric) |

---

## Workflow

1. **Input Aggregation** — Retrieved document chunks are concatenated into a single context block, preserving their order and metadata.
2. **Mode Selection** — Based on the `summary_mode` parameter, the skill chooses between abstractive (LLM-driven rewriting) or extractive (sentence scoring via TF-IDF or BERT) summarization.
3. **Query-Focused Filtering** *(optional)* — If a query is provided, the skill prioritizes sentences and paragraphs semantically aligned with the query before summarization.
4. **LLM Summarization** — For abstractive mode, the aggregated content and an instruction prompt are sent to the LLM (e.g., Gemini 1.5 Flash, GPT-4o) to generate a fluent summary.
5. **Key Point Extraction** — The summary is post-processed to extract bullet-point highlights for quick scanning.
6. **Compression Reporting** — The token reduction ratio is computed and logged for performance monitoring.
7. **Output Packaging** — The summary, key points, and metrics are returned as a structured response object.

---

## Example Input

```python
document_chunks = [
    "The transformer architecture was introduced in 2017 by Vaswani et al. in 'Attention is All You Need'...",
    "Unlike RNNs, transformers process sequences in parallel using self-attention mechanisms...",
    "The encoder maps input tokens to contextual embeddings while the decoder generates output tokens auto-regressively...",
    "BERT, GPT, and T5 are among the most influential transformer-based models developed after the original paper..."
]
query = "How does the transformer architecture work?"
max_summary_length = 150
summary_mode = "abstractive"
```

---

## Example Output

```json
{
  "summary": "The transformer architecture, introduced in 2017, revolutionized NLP by replacing recurrent networks with self-attention mechanisms that process sequences in parallel. It consists of an encoder that generates contextual embeddings and a decoder that produces outputs auto-regressively. This design enabled the development of landmark models like BERT, GPT, and T5, which form the backbone of modern language AI.",
  "key_points": [
    "Introduced in 2017 by Vaswani et al. in 'Attention is All You Need'",
    "Uses self-attention instead of recurrence for parallel sequence processing",
    "Encoder-decoder structure for embedding and generation",
    "Foundation for BERT, GPT, T5, and most modern LLMs"
  ],
  "token_reduction_ratio": 0.31
}
```

---

## Use Cases

1. **Research Paper Briefings** — Summarize academic papers for students or researchers who need quick overviews before deciding whether to read in full.
2. **Legal Contract Review** — Condense lengthy contracts into clause-level summaries for faster review by legal teams.
3. **News Aggregation** — Generate concise summaries of multiple news articles on the same topic for daily briefings.
4. **Customer Support Ticket Analysis** — Summarize long support threads to give agents instant context before responding.
5. **Medical Record Summarization** — Condense patient history records into structured summaries for clinical decision support.
6. **Board Meeting Minutes** — Automatically summarize meeting transcripts into action items and key decisions.

---

## Benefits

- **Token Efficiency** — Reduces the context window consumption of downstream LLM calls, lowering API costs.
- **Improved Readability** — Converts dense, jargon-heavy documents into plain-language summaries accessible to non-experts.
- **Query-Guided Focus** — Produces targeted summaries aligned with user intent rather than generic overviews.
- **Dual-Mode Flexibility** — Extractive mode preserves original wording for legal/compliance use; abstractive mode generates natural prose.
- **Audit Trail** — Key points enable rapid human verification of summarized content without reading the full output.

---

## Future Enhancements

- **Hierarchical Summarization** — Summarize individual chunks first, then synthesize into a global summary (MapReduce chain) for very long documents.
- **Multi-Document Synthesis** — Merge and reconcile information across multiple documents into a unified summary.
- **Structured Output Formatting** — Generate summaries in domain-specific formats (e.g., SOAP notes for medical, IRAC structure for legal).
- **Summarization Quality Scoring** — Integrate ROUGE / BERTScore metrics to evaluate summary quality automatically.
- **Streaming Summaries** — Support token-by-token streaming output for real-time UI display.

---

## Dependencies

| Library / Tool | Purpose |
|----------------|---------|
| `langchain` | LLM chains, prompt templates, MapReduce summarization |
| `google-generativeai` | Gemini 1.5 Flash / Pro for abstractive summarization |
| `transformers` | BERT-based extractive summarization models |
| `sumy` | Python library for extractive summarization algorithms |
| `tiktoken` | Token counting for length management |
| `rouge-score` | Evaluation metric for summarization quality |

---

> **Maintainer:** Team Trident | GITAM University, Hyderabad
> **Project:** RAG Chatbot — Algonive Internship 2024–25
> **License:** MIT
