---
title: "Method Comparisons"
---

# 📊 Comparison of Different Methods

Systematic comparison of retrievers, rerankers, and RAG methods.

## Retriever Comparison

```python
from rankify.dataset.dataset import Dataset
from rankify.metrics.metrics import Metrics

retrievers = ["bm25", "dpr", "contriever", "colbert"]
results = {}

for ret in retrievers:
    dataset = Dataset(retriever=ret, dataset_name="nq-dev", n_docs=100)
    docs = dataset.download()
    metrics = Metrics(docs)
    results[ret] = metrics.calculate_retrieval_metrics(ks=[5, 10])

import pandas as pd
df = pd.DataFrame(results).T
print(df)
```

## Reranker Comparison

```python
rerankers = {
    "monot5": ("monot5", "monot5-base-msmarco"),
    "flashrank": ("flashrank", "ms-marco-MiniLM-L-12-v2"),
}

for name, (method, model) in rerankers.items():
    reranker = Reranking(method=method, model_name=model)
    reranked = reranker.rank(documents)
    metrics = Metrics(reranked)
    results[name] = metrics.calculate_retrieval_metrics(ks=[5, 10], use_reordered=True)
```

## Creating a Report

```python
# Combine all results into a comprehensive table
print(pd.DataFrame(results).T.to_markdown())
```
