# Document Search Skill

## Overview

The **Document Search Skill** is a core retrieval component of the RAG (Retrieval-Augmented Generation) chatbot pipeline. It enables the agent to semantically search through a vector store of indexed documents and retrieve the most contextually relevant chunks in response to a user query. Unlike traditional keyword-based search, this skill leverages dense vector embeddings to understand the semantic meaning behind queries and documents.

---

## Purpose

In a RAG chatbot, the quality of retrieved context directly determines the quality of generated answers. The Document Search Skill acts as the **information retrieval backbone** of the pipeline. Without accurate and relevant document retrieval, the language model would be limited to its parametric knowledge, often producing hallucinated or outdated responses. This skill bridges the gap between raw user queries and a grounded, factual knowledge base.

---

## Inputs

| Input | Type | Description |
|-------|------|-------------|
| `query` | `str` | The natural language question or search phrase from the user |
| `vector_store` | `FAISS / ChromaDB index` | Pre-built vector index of embedded document chunks |
| `embedding_model` | `HuggingFace / OpenAI model` | The embedding model used to encode the query |
| `top_k` | `int` | Number of top relevant chunks to retrieve (default: 5) |
| `score_threshold` | `float` | Minimum similarity score to filter out irrelevant results (optional) |

---

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `retrieved_chunks` | `List[str]` | Top-k most relevant document chunks |
| `source_metadata` | `List[dict]` | Metadata for each chunk (filename, page number, chunk ID) |
| `similarity_scores` | `List[float]` | Cosine similarity scores for each retrieved chunk |

---

## Workflow

1. **Query Encoding** — The user's natural language query is passed through the embedding model to produce a dense query vector.
2. **Vector Store Search** — The query vector is compared against all stored document vectors using cosine similarity (or L2 distance) via FAISS or ChromaDB.
3. **Top-K Retrieval** — The `top_k` most similar document chunks are selected based on similarity score rankings.
4. **Score Filtering** — If a `score_threshold` is set, chunks below the threshold are discarded to maintain retrieval quality.
5. **Metadata Enrichment** — Each retrieved chunk is paired with its source metadata (document name, page number, section) for traceability.
6. **Context Assembly** — The retrieved chunks are packaged into a structured context object to be passed downstream to the generation skill.

---

## Example Input

```python
query = "What are the side effects of Metformin in diabetic patients?"
top_k = 5
score_threshold = 0.70
```

---

## Example Output

```json
{
  "retrieved_chunks": [
    "Metformin is generally well-tolerated but common side effects include nausea, diarrhea, and stomach upset, especially when first starting the medication...",
    "In rare cases, Metformin can cause lactic acidosis, a serious condition that requires immediate medical attention...",
    "Patients with renal impairment should exercise caution as Metformin clearance is reduced, increasing the risk of adverse effects..."
  ],
  "source_metadata": [
    {"source": "diabetes_treatment_guide.pdf", "page": 14, "chunk_id": "chunk_047"},
    {"source": "pharmacology_reference.pdf", "page": 82, "chunk_id": "chunk_213"},
    {"source": "clinical_notes_2023.pdf", "page": 7, "chunk_id": "chunk_091"}
  ],
  "similarity_scores": [0.91, 0.87, 0.83]
}
```

---

## Use Cases

1. **Medical Knowledge Base** — Retrieve relevant clinical guidelines or drug information in response to doctor or patient queries.
2. **Legal Document Search** — Search through case laws, contracts, or regulations to find applicable clauses.
3. **Corporate FAQ Automation** — Enable employees to search internal HR policies, compliance documents, or SOPs via natural language.
4. **Academic Research Assistant** — Retrieve relevant passages from a library of research papers based on a student's query.
5. **E-Commerce Product Search** — Find semantically matching product descriptions based on user intent rather than exact keyword matches.
6. **Customer Support** — Retrieve the most relevant help articles and troubleshooting steps from a product knowledge base.

---

## Benefits

- **Semantic Understanding** — Captures intent beyond exact keyword matching, retrieving results even when phrasing differs.
- **Scalable Retrieval** — FAISS enables sub-second search across millions of document chunks.
- **Source Traceability** — Metadata linkage ensures every retrieved chunk can be traced back to its origin document.
- **Modular Design** — Easily swappable with different vector stores (FAISS, ChromaDB, Pinecone, Weaviate) without changing downstream components.
- **Quality Filtering** — Score thresholds prevent noisy or irrelevant context from degrading answer generation.

---

## Future Enhancements

- **Hybrid Search** — Combine dense vector search with BM25 sparse retrieval (Reciprocal Rank Fusion) for improved recall on rare terms.
- **Re-ranking Layer** — Add a cross-encoder re-ranker (e.g., `ms-marco-MiniLM`) to re-score retrieved chunks for higher precision.
- **Multi-Index Support** — Route queries to domain-specific vector indexes (e.g., medical vs. legal) based on query classification.
- **Query Expansion** — Use LLM-generated query variants to improve retrieval coverage (HyDE — Hypothetical Document Embeddings).
- **Adaptive Top-K** — Dynamically adjust `top_k` based on query complexity and confidence scores.

---

## Dependencies

| Library / Tool | Purpose |
|----------------|---------|
| `langchain` | Document loaders, text splitters, retriever abstractions |
| `faiss-cpu` / `faiss-gpu` | Efficient vector similarity search |
| `sentence-transformers` | HuggingFace embedding models |
| `chromadb` | Lightweight persistent vector store (alternative) |
| `numpy` | Vector operations and similarity computations |
| `tiktoken` | Token counting for chunk size management |

---

> **Maintainer:** Team Trident | GITAM University, Hyderabad
> **Project:** RAG Chatbot — Algonive Internship 2024–25
> **License:** MIT
