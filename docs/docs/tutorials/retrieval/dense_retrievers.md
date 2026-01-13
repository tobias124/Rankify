---
title: "Dense Retrievers"
---

# 🧠 Using Dense Retrievers

Dense retrievers use neural networks to encode queries and documents into dense vector representations, enabling semantic similarity search.

## Overview

Rankify supports these dense retrieval methods:

| Method | Model | Best For |
|--------|-------|----------|
| DPR | Facebook DPR | General QA tasks |
| ANCE | Microsoft ANCE | High-precision retrieval |
| ColBERT | ColBERT v2 | Fine-grained matching |
| BGE | BAAI BGE | Multilingual, general use |
| Contriever | Meta Contriever | Zero-shot retrieval |

## DPR (Dense Passage Retrieval)

```python
from rankify.dataset.dataset import Document, Question
from rankify.retrievers.retriever import Retriever

documents = [Document(question=Question("Who discovered penicillin?"))]

# DPR Multi-encoder (recommended)
dpr_retriever = Retriever(
    method="dpr-multi",
    n_docs=10,
    index_type="wiki"
)
results = dpr_retriever.retrieve(documents)

# DPR Single-encoder (faster)
dpr_single = Retriever(
    method="dpr-single",
    n_docs=10,
    index_type="wiki"
)
```

## ANCE (Approximate Nearest Neighbor Contrastive Estimation)

ANCE uses hard negative mining for improved retrieval quality:

```python
ance_retriever = Retriever(
    method="ance-multi",
    n_docs=10,
    index_type="wiki"
)
results = ance_retriever.retrieve(documents)
```

## ColBERT (Contextualized Late Interaction)

ColBERT provides fine-grained token-level matching:

```python
colbert_retriever = Retriever(
    method="colbert",
    n_docs=10,
    index_type="wiki"
)
results = colbert_retriever.retrieve(documents)
```

!!! note "ColBERT Setup"
    ColBERT requires additional setup. See [Installation Guide](../../installation.md).

## BGE (BAAI General Embedding)

BGE offers strong performance across multiple languages:

```python
bge_retriever = Retriever(
    method="bge",
    n_docs=10,
    index_type="wiki"
)
results = bge_retriever.retrieve(documents)
```

## Contriever

Meta's contrastive retriever excels at zero-shot retrieval:

```python
contriever_retriever = Retriever(
    method="contriever",
    n_docs=10,
    index_type="wiki"
)
results = contriever_retriever.retrieve(documents)
```

## Building Custom Dense Indices

Use the CLI to build indices for your own corpus:

```bash
# DPR Index
rankify-index index data/corpus.jsonl \
    --retriever dpr \
    --encoder facebook/dpr-ctx_encoder-single-nq-base \
    --batch_size 16 --device cuda

# ANCE Index
rankify-index index data/corpus.jsonl \
    --retriever ance \
    --encoder castorini/ance-dpr-context-multi \
    --batch_size 16 --device cuda

# BGE Index
rankify-index index data/corpus.jsonl \
    --retriever bge \
    --encoder BAAI/bge-large-en-v1.5 \
    --batch_size 16 --device cuda

# Contriever Index
rankify-index index data/corpus.jsonl \
    --retriever contriever \
    --encoder facebook/contriever-msmarco \
    --batch_size 16 --device cuda

# ColBERT Index
rankify-index index data/corpus.jsonl \
    --retriever colbert \
    --batch_size 32 --device cuda
```

## Comparison of Dense Retrievers

| Method | Speed | Accuracy | Memory | Zero-shot |
|--------|-------|----------|--------|-----------|
| DPR | Fast | Good | Medium | No |
| ANCE | Medium | Very Good | Medium | No |
| ColBERT | Slow | Excellent | High | No |
| BGE | Fast | Very Good | Medium | Yes |
| Contriever | Fast | Good | Medium | Yes |

## Next Steps

- [🤖 Hybrid Retrieval](hybrid.md) - Combine sparse and dense methods
- [📂 Prebuilt Corpora](prebuilt_corpora.md) - Available pre-indexed datasets
- [📊 Reranking](../reranking/introduction.md) - Improve dense retrieval with reranking
