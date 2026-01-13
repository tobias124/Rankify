---
title: "Saving & Loading Models"
---

# 💾 Saving & Loading Models

Manage model caching and persistence.

## Cache Directory

```python
import os

# Default cache
print(os.environ.get("RERANKING_CACHE_DIR", "~/.cache/rankify"))

# Custom cache location
os.environ["RERANKING_CACHE_DIR"] = "/path/to/cache"
```

## Model Caching

Models are automatically cached on first use:

```python
# First call downloads and caches the model
reranker = Reranking(method="monot5", model_name="monot5-base-msmarco")

# Subsequent calls use cached version
reranker2 = Reranking(method="monot5", model_name="monot5-base-msmarco")
```

## Saving Reranked Results

```python
from rankify.dataset.dataset import Dataset

# Save documents with reranked contexts
Dataset.save_dataset(reranked_documents, "./reranked_results.json")
```

## Loading Saved Results

```python
# Load previously saved results
documents = Dataset.load_dataset("./reranked_results.json", n_docs=100)
```
