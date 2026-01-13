---
title: "Introduction to Information Retrieval"
---

# 📌 Introduction to Information Retrieval

Information Retrieval (IR) is the foundation of search systems, question answering, and RAG pipelines. Rankify provides a unified interface for multiple retrieval methods.

## What is Information Retrieval?

Information retrieval is the task of finding documents that are relevant to a user's query from a large collection. In Rankify, this involves:

1. **Indexing**: Building searchable indices from document collections
2. **Querying**: Finding relevant documents for a given question
3. **Ranking**: Ordering results by relevance

## Retrieval Methods in Rankify

Rankify supports **10 retrieval methods**:

| Method | Type | Description |
|--------|------|-------------|
| BM25 | Sparse | Classic term-matching algorithm |
| DPR | Dense | Facebook's Dense Passage Retrieval |
| ANCE | Dense | Approximate Nearest Neighbor Contrastive Estimation |
| BGE | Dense | BAAI General Embedding |
| ColBERT | Dense | Late interaction with contextualized embeddings |
| Contriever | Dense | Contrastive retriever from Meta |
| Online | Web | Real-time web search |
| HyDE | Dense | Hypothetical Document Embeddings |

## Quick Start

```python
from rankify.dataset.dataset import Document, Question
from rankify.retrievers.retriever import Retriever

# Create a query
documents = [Document(question=Question("Who wrote Hamlet?"))]

# Initialize a retriever
retriever = Retriever(method="bm25", n_docs=5, index_type="wiki")

# Retrieve relevant passages
retrieved_docs = retriever.retrieve(documents)

# Print results
for doc in retrieved_docs:
    for ctx in doc.contexts:
        print(f"[{ctx.score:.4f}] {ctx.text[:100]}...")
```

## Choosing a Retrieval Method

| Use Case | Recommended Method |
|----------|-------------------|
| Fast, efficient search | BM25 |
| Semantic understanding | DPR, BGE, Contriever |
| High accuracy | ColBERT, ANCE |
| Real-time web data | Online |
| Zero-shot retrieval | HyDE |

## Next Steps

- [🔍 Using BM25](bm25.md) - Learn sparse retrieval
- [🧠 Dense Retrievers](dense_retrievers.md) - DPR, ANCE, ColBERT, BGE, Contriever
- [📂 Prebuilt Corpora](prebuilt_corpora.md) - Use pre-indexed Wikipedia and MS MARCO
