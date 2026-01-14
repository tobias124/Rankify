---
title: "Hybrid Retrieval - BM25 + Dense Fusion"
---

# 🔀 Hybrid Retrieval

Combine sparse (BM25) and dense retrieval for state-of-the-art results using Reciprocal Rank Fusion (RRF).

## Quick Start

```python
from rankify.retrievers.hybrid_retriever import HybridRetriever

# Combine BM25 + BGE with RRF fusion
retriever = HybridRetriever(
    sparse="bm25",
    dense="bge",
    fusion="rrf",
)

documents = retriever.retrieve(documents)
```

---

## Why Hybrid Retrieval?

| Retrieval Type | Strengths | Weaknesses |
|----------------|-----------|------------|
| **Sparse (BM25)** | Exact matches, keywords | Misses semantic meaning |
| **Dense (BGE, DPR)** | Semantic understanding | May miss exact keywords |
| **Hybrid** | Best of both! | Slightly slower |

---

## Fusion Strategies

### 1. Reciprocal Rank Fusion (RRF) - Recommended

```python
from rankify.retrievers.hybrid_retriever import HybridRetriever

retriever = HybridRetriever(
    sparse="bm25",
    dense="bge",
    fusion="rrf",
    rrf_k=60,  # Higher = more weight on top ranks
)
```

**How RRF Works:**
```
RRF_score(doc) = sum(1 / (k + rank_i)) for each retriever
```

### 2. Weighted Combination

```python
retriever = HybridRetriever(
    sparse="bm25",
    dense="bge",
    fusion="weighted",
    weights=[0.3, 0.7],  # 30% BM25, 70% BGE
)
```

### 3. Interleave

```python
retriever = HybridRetriever(
    sparse="bm25",
    dense="bge",
    fusion="interleave",  # Alternate results
)
```

---

## Multiple Dense Retrievers

Combine BM25 with multiple dense retrievers:

```python
retriever = HybridRetriever(
    sparse="bm25",
    dense=["dpr", "bge", "colbert"],  # Multiple dense
    fusion="rrf",
    weights=[0.2, 0.3, 0.3, 0.2],  # Optional weights
)
```

---

## Complete Example

```python
from rankify.retrievers.hybrid_retriever import HybridRetriever
from rankify.dataset.dataset import Document, Question, Answer

# Create hybrid retriever
retriever = HybridRetriever(
    sparse="bm25",
    dense="bge",
    fusion="rrf",
    n_docs=100,
)

# Prepare query
doc = Document(
    question=Question("What are transformers in NLP?"),
    answers=Answer([]),
    contexts=[],
)

# Retrieve with hybrid fusion
results = retriever.retrieve([doc])

# Get fused results
print(f"Retrieved {len(results[0].contexts)} documents")
for ctx in results[0].contexts[:5]:
    print(f"Score: {ctx.score:.3f} - {ctx.text[:50]}...")
```

**Expected Output:**
```
Retrieved 100 documents
Score: 0.0323 - Transformers are a type of neural network architecture...
Score: 0.0312 - The transformer model was introduced in "Attention Is All You Need"...
Score: 0.0298 - BERT uses the transformer encoder for NLP tasks...
Score: 0.0287 - GPT models are based on the transformer decoder...
Score: 0.0275 - Self-attention is the key mechanism in transformers...
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sparse` | `"bm25"` | Sparse retriever method |
| `dense` | `"bge"` | Dense retriever(s) |
| `fusion` | `"rrf"` | Fusion strategy |
| `weights` | Equal | Weights for each retriever |
| `n_docs` | `100` | Documents per retriever |
| `rrf_k` | `60` | RRF parameter |

---

## Best Practices

1. **Start with RRF** - Works well without tuning
2. **Use BGE for dense** - Best balance of speed and accuracy
3. **Tune weights for your domain** - Test with your data
4. **Consider ColBERT for precision** - Best accuracy but slower
