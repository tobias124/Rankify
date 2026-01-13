---
title: "Debugging"
---

# 🐞 Debugging & Observability

Debug and monitor Rankify applications.

## Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rankify")
```

## Inspecting Results

```python
def debug_retrieval(documents):
    """Inspect retrieval results."""
    for doc in documents:
        print(f"Q: {doc.question.question}")
        print(f"Contexts retrieved: {len(doc.contexts)}")
        for i, ctx in enumerate(doc.contexts[:3]):
            print(f"  [{i}] Score: {ctx.score:.4f}")
            print(f"      Has answer: {ctx.has_answer}")
```

## Performance Profiling

```python
import time

def timed_operation(name, func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.2f}s")
    return result

# Usage
reranked = timed_operation("Reranking", reranker.rank, documents)
```

## Memory Tracking

```python
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```
