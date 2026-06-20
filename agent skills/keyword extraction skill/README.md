# Keyword Extraction Skill

## Overview

The **Keyword Extraction Skill** identifies and extracts the most semantically significant terms, phrases, and named entities from user queries, retrieved document chunks, and generated answers. Operating as both a pre-retrieval query enhancer and a post-generation indexing utility, this skill uses a combination of statistical methods (TF-IDF, YAKE), graph-based algorithms (TextRank), and transformer-based models (KeyBERT) to produce high-quality, context-aware keyword sets. These keywords serve multiple roles across the RAG pipeline — from improving search precision to enabling document tagging and topic modeling.

---

## Purpose

Keywords are the connective tissue between user intent and knowledge base content. In a RAG chatbot, poorly formed queries often fail to retrieve the most relevant documents because they lack the precise terminology present in the indexed corpus. The Keyword Extraction Skill solves this by expanding and enriching queries with domain-relevant terms before retrieval. It also enables **automatic document tagging**, **topic clustering**, and **faceted search** on the knowledge base, making the entire retrieval ecosystem more robust and discoverable.

---

## Inputs

| Input | Type | Description |
|-------|------|-------------|
| `text` | `str` | Input text — can be a user query, document chunk, or generated answer |
| `extraction_method` | `str` | Algorithm to use: `"keybert"`, `"yake"`, `"tfidf"`, `"textrank"` |
| `top_n` | `int` | Number of top keywords/phrases to return (default: 10) |
| `ngram_range` | `tuple` | Min and max n-gram length, e.g., `(1, 3)` for unigrams to trigrams |
| `diversity` | `float` | MMR diversity parameter for KeyBERT (0 = redundant, 1 = diverse; default: 0.5) |
| `language` | `str` | Language of the input text for multilingual support (default: `"en"`) |

---

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `keywords` | `List[str]` | Ranked list of extracted keywords and key phrases |
| `scores` | `List[float]` | Relevance or importance score for each keyword |
| `named_entities` | `dict` | Categorized named entities: `PERSON`, `ORG`, `DATE`, `LOCATION`, `TECH` |
| `keyword_cloud_data` | `dict` | Keyword-frequency mapping suitable for word cloud visualization |

---

## Workflow

1. **Text Preprocessing** — Input text is lowercased, punctuation-stripped, and stop words are removed. Domain-specific stop words (e.g., "the document states") are filtered using a custom stop list.
2. **Method Selection** — Based on the `extraction_method` parameter, the appropriate algorithm is initialized:
   - **KeyBERT**: Embeds the text and candidate n-grams; selects candidates most similar to the full document embedding using Maximal Marginal Relevance (MMR).
   - **YAKE**: Uses statistical features (word frequency, position, co-occurrence) to score candidates without requiring a language model.
   - **TF-IDF**: Scores terms based on frequency within the document relative to a background corpus.
   - **TextRank**: Builds a word co-occurrence graph and applies PageRank to score nodes.
3. **N-Gram Candidate Generation** — Candidate unigrams, bigrams, and trigrams are generated from the preprocessed text based on the `ngram_range` parameter.
4. **Scoring & Ranking** — Each candidate is scored by the selected algorithm and sorted in descending order of importance.
5. **Named Entity Recognition (NER)** — SpaCy's NER pipeline is applied to extract and categorize named entities independently of the keyword ranking.
6. **Deduplication & Diversity** — Redundant keywords (subsets of longer phrases already extracted) are pruned. MMR is applied in KeyBERT mode to ensure diverse coverage.
7. **Output Assembly** — Keywords, scores, named entities, and word cloud data are packaged and returned.

---

## Example Input

```python
text = """
Deep learning models have transformed natural language processing tasks such as machine translation,
sentiment analysis, and named entity recognition. Pre-trained transformer models like BERT and GPT-4
have achieved state-of-the-art results across multiple NLP benchmarks including GLUE and SuperGLUE.
Fine-tuning these models on domain-specific datasets remains a key technique for applied NLP.
"""

extraction_method = "keybert"
top_n = 8
ngram_range = (1, 2)
diversity = 0.6
```

---

## Example Output

```json
{
  "keywords": [
    "transformer models",
    "natural language processing",
    "fine-tuning",
    "BERT",
    "GPT-4",
    "NLP benchmarks",
    "deep learning",
    "sentiment analysis"
  ],
  "scores": [0.94, 0.91, 0.87, 0.85, 0.83, 0.79, 0.76, 0.72],
  "named_entities": {
    "TECH": ["BERT", "GPT-4", "GLUE", "SuperGLUE"],
    "TASK": ["machine translation", "sentiment analysis", "named entity recognition"]
  },
  "keyword_cloud_data": {
    "transformer models": 3,
    "BERT": 2,
    "fine-tuning": 2,
    "natural language processing": 2,
    "GPT-4": 1
  }
}
```

---

## Use Cases

1. **Query Expansion** — Automatically enrich short or vague user queries with domain-relevant synonyms and related terms before vector search, improving retrieval recall.
2. **Document Auto-Tagging** — Assign keyword tags to newly ingested documents for faceted browsing and filtered search in the knowledge base UI.
3. **Topic Modeling Dashboard** — Cluster documents by shared keywords to generate topic maps for knowledge base exploration.
4. **SEO Optimization for Knowledge Bases** — Extract and index keywords from corporate knowledge bases to improve internal search engine discoverability.
5. **Chatbot Intent Classification** — Use extracted keywords as features for a lightweight intent classifier to route queries to specialized sub-agents.
6. **Content Recommendation** — Recommend related documents by comparing keyword overlap between the current document and the rest of the corpus.

---

## Benefits

- **Retrieval Enhancement** — Extracted keywords feed back into the search pipeline as query expansion terms, closing the vocabulary mismatch gap.
- **Language Model Agnostic** — YAKE and TF-IDF methods work without any LLM, making the skill fast, lightweight, and deployable in low-resource environments.
- **Semantic Depth via KeyBERT** — Embedding-based extraction captures contextually meaningful multi-word phrases beyond simple frequency metrics.
- **NER Integration** — Named entity extraction adds a structured layer of understanding — especially valuable in technical, medical, and legal domains.
- **Visualization Ready** — Word cloud data output enables instant visual analytics without additional preprocessing.

---

## Future Enhancements

- **Domain-Adaptive Keyword Extraction** — Fine-tune KeyBERT on domain corpora (medical, legal, financial) to extract field-specific terminology more accurately.
- **Cross-Document Keyword Comparison** — Compute keyword overlap matrices across document sets for plagiarism detection or literature gap analysis.
- **Temporal Keyword Tracking** — Track keyword frequency over time across ingested documents to detect emerging topics and trends.
- **Keyphrase to Query Reformulation** — Use extracted keyphrases to automatically reformulate ambiguous queries into precise, structured retrieval queries.
- **Multilingual Support** — Extend keyword extraction to 50+ languages using multilingual sentence-transformers and language-agnostic YAKE.

---

## Dependencies

| Library / Tool | Purpose |
|----------------|---------|
| `keybert` | Embedding-based keyword and keyphrase extraction |
| `yake` | Statistical keyword extraction without language model dependency |
| `scikit-learn` | TF-IDF vectorization and corpus-level keyword scoring |
| `spacy` | Named Entity Recognition (NER) and linguistic preprocessing |
| `networkx` | TextRank graph construction and PageRank scoring |
| `nltk` | Stop word filtering and tokenization utilities |
| `sentence-transformers` | Embedding backbone for KeyBERT (`all-MiniLM-L6-v2`) |
| `wordcloud` | Optional word cloud visualization from keyword-frequency data |

---

> **Maintainer:** Team Trident | GITAM University, Hyderabad
> **Project:** RAG Chatbot — Algonive Internship 2024–25
> **License:** MIT
