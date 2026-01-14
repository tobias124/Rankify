---
title: "Pipeline API - Simple One-Line Interface"
---

# 🚀 Pipeline API

The Pipeline API provides the simplest way to use Rankify with a HuggingFace-style one-line interface.

## Quick Start

```python
from rankify import pipeline

# Create a RAG pipeline with one line
rag = pipeline("rag")
answers = rag("What is machine learning?", documents)

print(answers)
```

**Expected Output:**
```
PipelineResult(query='What is machine learning?...', answers=1, docs=100)
```

---

## Pipeline Types

| Type | Description | Components |
|------|-------------|------------|
| `"search"` | Document retrieval only | Retriever |
| `"rerank"` | Retrieve + rerank | Retriever + Reranker |
| `"rag"` | Full RAG pipeline | Retriever + Reranker + Generator |

### Search Pipeline

```python
from rankify import pipeline

search = pipeline("search", retriever="bm25", n_docs=100)
results = search("Find papers about transformers")

for ctx in results.documents[0].contexts[:5]:
    print(f"- {ctx.text[:100]}...")
```

**Expected Output:**
```
- Attention Is All You Need introduces the Transformer architecture...
- BERT: Pre-training of Deep Bidirectional Transformers...
- GPT-3: Language Models are Few-Shot Learners...
```

### Rerank Pipeline

```python
from rankify import pipeline

rerank = pipeline("rerank", retriever="bge", reranker="flashrank")
results = rerank("Best Python frameworks for ML")

for ctx in results.documents[0].reorder_contexts[:5]:
    print(f"Score: {ctx.score:.3f} - {ctx.text[:50]}...")
```

**Expected Output:**
```
Score: 0.892 - PyTorch is a popular deep learning framework...
Score: 0.856 - TensorFlow provides comprehensive ML tools...
Score: 0.823 - Scikit-learn offers simple ML algorithms...
```

### RAG Pipeline

```python
from rankify import pipeline

rag = pipeline(
    "rag",
    retriever="bge",
    reranker="monot5",
    generator="chain-of-thought-rag",
)

result = rag("Explain how transformers work")
print(result.answers[0])
```

**Expected Output:**
```
Transformers work through a self-attention mechanism that allows 
the model to weigh the importance of different parts of the input...
```

---

## Configuration Options

```python
from rankify import pipeline

rag = pipeline(
    task="rag",                          # Pipeline type
    retriever="bge",                     # Retriever method
    reranker="flashrank",                # Reranker method
    generator="basic-rag",               # RAG method
    retriever_model="BAAI/bge-base",     # Specific retriever model
    reranker_model="ms-marco-MiniLM",    # Specific reranker model
    generator_model="gpt-4o-mini",       # LLM model
    generator_backend="openai",          # LLM backend
    n_docs=100,                          # Docs to retrieve
    top_k=5,                             # Docs for generation
    index_type="wiki",                   # Index type
)
```

---

## Convenience Functions

```python
from rankify import search_pipeline, rerank_pipeline, rag_pipeline

# Quick search
search = search_pipeline(retriever="bm25")

# Quick rerank
rerank = rerank_pipeline(retriever="bge", reranker="flashrank")

# Quick RAG
rag = rag_pipeline(retriever="bge", reranker="monot5", generator="basic-rag")
```

---

## Available Methods

### Retrievers
- `bm25` - Fast sparse retrieval (default)
- `dpr` - Dense Passage Retrieval
- `bge` - BAAI General Embedding
- `ance` - ANCE retriever
- `colbert` - ColBERT retriever
- `contriever` - Contriever retriever

### Rerankers
- `flashrank` - Ultra-fast ONNX reranker (default)
- `monot5` - MonoT5 reranker
- `rankgpt` - LLM-based RankGPT
- `inranker` - InRanker
- `colbert_ranker` - ColBERT reranker
- `upr` - UPR reranker

### RAG Methods
- `basic-rag` - Basic RAG (default)
- `chain-of-thought-rag` - CoT reasoning
- `self-consistency-rag` - Multiple generations
- `zero-shot` - Zero-shot generation

---

## Pipeline Methods

```python
from rankify import pipeline

rag = pipeline("rag")

# Main call
result = rag("Your question", documents)

# Or use run()
result = rag.run("Your question", documents)

# Just retrieve
contexts = rag.retrieve("Query", n_docs=50)

# Just rerank
reranked = rag.rerank("Query", contexts, top_k=10)

# Just generate
answer = rag.generate("Query", contexts)
```

---

## Next Steps

- [Hybrid Retrieval](hybrid_retrieval.md) - Combine BM25 + Dense
- [Rankify Server](server.md) - Deploy as REST API
- [Integrations](integrations.md) - LangChain & LlamaIndex
