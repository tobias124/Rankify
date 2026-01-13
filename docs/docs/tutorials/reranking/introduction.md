---
title: "Introduction to Re-Ranking"
---

# 📌 Introduction to Re-Ranking

Re-ranking improves retrieval results by reordering documents using more sophisticated models.

## What is Re-Ranking?

Re-ranking is a two-stage approach:
1. **Stage 1 (Retrieval)**: Fast retrieval of top-k candidates (e.g., BM25)
2. **Stage 2 (Re-ranking)**: Neural model reorders candidates for better relevance

## Re-Ranking Methods in Rankify

Rankify supports **23 re-ranking methods** across different paradigms:

### Pointwise Rerankers
Score each query-document pair independently:

| Method | Model Type |
|--------|------------|
| MonoBERT | BERT cross-encoder |
| MonoT5 | T5 sequence-to-sequence |
| UPR | Unsupervised passage reranker |

### Pairwise Rerankers
Compare document pairs:

| Method | Description |
|--------|-------------|
| RankGPT | LLM-based pairwise ranking |
| InRanker | Instruction-based reranking |
| EchoRank | Echo-based pairwise comparison |

### Listwise Rerankers
Consider entire document list:

| Method | Description |
|--------|-------------|
| RankT5 | T5-based listwise ranking |
| ListT5 | Listwise T5 model |
| LiT5 | Lightweight T5 reranker |

### API-Based Rerankers
External API services:

| Provider | Description |
|----------|-------------|
| Cohere | Cohere Rerank API |
| Jina | Jina Reranker API |
| Voyage | Voyage Rerank API |
| MixedBread | MixedBread.ai API |

## Quick Start

```python
from rankify.dataset.dataset import Document, Question, Context
from rankify.models.reranking import Reranking

# Create a document with retrieved contexts
question = Question("When did Einstein win the Nobel Prize?")
contexts = [
    Context(text="Einstein received the Nobel Prize in Physics in 1921.", id="1"),
    Context(text="Albert Einstein was born in Germany in 1879.", id="2"),
    Context(text="The Nobel Prize is awarded annually in Stockholm.", id="3"),
]
document = Document(question=question, contexts=contexts)

# Initialize a reranker
reranker = Reranking(method="monot5", model_name="monot5-base-msmarco")

# Rerank the documents
reranked_docs = reranker.rank([document])

# Access reordered contexts
for ctx in reranked_docs[0].reorder_contexts:
    print(f"[{ctx.score:.4f}] {ctx.text}")
```

## Choosing a Reranker

| Use Case | Recommended |
|----------|-------------|
| Fast, accurate | MonoT5, FlashRank |
| Best quality | RankGPT, ColBERT Reranker |
| No GPU | API Rerankers (Cohere, Jina) |
| Large batches | Transformer Reranker |
| LLM-based | RankGPT, Vicuna, Zephyr |

## Next Steps

- [🎯 Pointwise Reranking](pointwise.md) - MonoBERT, MonoT5
- [🔄 Pairwise Reranking](pairwise.md) - RankGPT, InRanker
- [📃 Listwise Reranking](listwise.md) - RankT5, LiT5
- [🦾 API Rerankers](api_rerankers.md) - Cohere, Jina, Voyage
