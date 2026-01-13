---
title: "Reranking Metrics"
---

# 📈 Evaluating Rerankers

Measure reranking improvement with standard IR metrics.

## Key Metrics

| Metric | Description |
|--------|-------------|
| NDCG@k | Normalized Discounted Cumulative Gain |
| MAP | Mean Average Precision |
| MRR | Mean Reciprocal Rank |

## Before vs After Reranking

```python
from rankify.metrics.metrics import Metrics
from rankify.models.reranking import Reranking

# Before reranking
metrics = Metrics(documents)
before = metrics.calculate_retrieval_metrics(ks=[1, 5, 10], use_reordered=False)
print("Before:", before)

# After reranking
reranker = Reranking(method="monot5", model_name="monot5-base-msmarco")
reranked = reranker.rank(documents)

metrics = Metrics(reranked)
after = metrics.calculate_retrieval_metrics(ks=[1, 5, 10], use_reordered=True)
print("After:", after)
```

## Improvement Analysis

```python
improvement = {
    k: after[k] - before[k] 
    for k in before.keys() 
    if k.startswith("Recall")
}
print("Improvement:", improvement)
```
