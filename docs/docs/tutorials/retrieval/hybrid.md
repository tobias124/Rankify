---
title: "Hybrid Retrieval"
---

# 🤖 Hybrid Retrieval (Combining Sparse & Dense)

Hybrid retrieval combines the strengths of sparse (BM25) and dense retrievers to achieve better retrieval performance.

## Why Hybrid Retrieval?

| Sparse (BM25) | Dense (DPR, etc.) |
|---------------|-------------------|
| ✅ Exact keyword matching | ✅ Semantic understanding |
| ✅ Fast and efficient | ✅ Handles synonyms |
| ❌ Misses synonyms | ❌ May miss exact terms |
| ❌ No semantic understanding | ❌ Computationally expensive |

**Hybrid combines both for better results!**

## Building a Hybrid Pipeline

```python
from rankify.dataset.dataset import Document, Question, Context
from rankify.retrievers.retriever import Retriever

# Create query
documents = [Document(question=Question("What causes climate change?"))]

# Step 1: BM25 retrieval
bm25_retriever = Retriever(method="bm25", n_docs=50, index_type="wiki")
bm25_results = bm25_retriever.retrieve(documents)

# Step 2: Dense retrieval
dense_retriever = Retriever(method="contriever", n_docs=50, index_type="wiki")
dense_results = dense_retriever.retrieve(documents.copy())  # Fresh copy

# Step 3: Merge results using Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(bm25_docs, dense_docs, k=60):
    """Combine rankings using RRF."""
    scores = {}
    
    for doc, dense_doc in zip(bm25_docs, dense_docs):
        # Process BM25 results
        for rank, ctx in enumerate(doc.contexts):
            if ctx.id not in scores:
                scores[ctx.id] = {"ctx": ctx, "score": 0}
            scores[ctx.id]["score"] += 1 / (k + rank + 1)
        
        # Process dense results
        for rank, ctx in enumerate(dense_doc.contexts):
            if ctx.id not in scores:
                scores[ctx.id] = {"ctx": ctx, "score": 0}
            scores[ctx.id]["score"] += 1 / (k + rank + 1)
    
    # Sort by combined score
    sorted_results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["ctx"] for item in sorted_results[:20]]

# Apply fusion
hybrid_contexts = reciprocal_rank_fusion(bm25_results, dense_results)

# Create final document with hybrid results
hybrid_doc = Document(
    question=documents[0].question,
    contexts=hybrid_contexts
)
```

## HyDE: Hypothetical Document Embeddings

HyDE is a zero-shot hybrid approach that generates hypothetical documents:

```python
# HyDE generates a hypothetical answer and uses it for retrieval
hyde_retriever = Retriever(
    method="hyde",
    n_docs=10,
    index_type="wiki"
)

documents = [Document(question=Question("How does photosynthesis work?"))]
results = hyde_retriever.retrieve(documents)
```

## Weighted Combination

For more control, use weighted score combination:

```python
def weighted_hybrid(bm25_ctxs, dense_ctxs, bm25_weight=0.3, dense_weight=0.7):
    """Weighted score combination."""
    all_contexts = {}
    
    # Normalize and weight BM25 scores
    if bm25_ctxs:
        max_bm25 = max(ctx.score for ctx in bm25_ctxs) or 1
        for ctx in bm25_ctxs:
            all_contexts[ctx.id] = {
                "ctx": ctx,
                "score": bm25_weight * (ctx.score / max_bm25)
            }
    
    # Add weighted dense scores
    if dense_ctxs:
        max_dense = max(ctx.score for ctx in dense_ctxs) or 1
        for ctx in dense_ctxs:
            if ctx.id in all_contexts:
                all_contexts[ctx.id]["score"] += dense_weight * (ctx.score / max_dense)
            else:
                all_contexts[ctx.id] = {
                    "ctx": ctx,
                    "score": dense_weight * (ctx.score / max_dense)
                }
    
    sorted_results = sorted(all_contexts.values(), key=lambda x: x["score"], reverse=True)
    return [item["ctx"] for item in sorted_results]
```

## Best Practices

1. **Retrieve more initially**: Get 50-100 docs from each method, then merge to top-k
2. **Tune weights**: Optimal BM25/dense weights depend on your domain
3. **Add reranking**: Hybrid + neural reranker often gives best results

## Next Steps

- [📂 Prebuilt Corpora](prebuilt_corpora.md) - Available indices
- [📊 Reranking](../reranking/introduction.md) - Improve hybrid results further
- [⚙️ RAG Pipelines](../rag/pipelines.md) - End-to-end systems
