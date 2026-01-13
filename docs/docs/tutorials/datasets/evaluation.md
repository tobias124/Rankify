---
title: "Dataset Evaluation"
---

# 📊 Evaluating Retrieval & Ranking on Datasets

Evaluate retrieval and reranking quality on standard datasets.

## Retrieval Evaluation

```python
from rankify.dataset.dataset import Dataset
from rankify.metrics.metrics import Metrics

dataset = Dataset(retriever="bm25", dataset_name="nq-dev", n_docs=100)
documents = dataset.download()

metrics = Metrics(documents)
results = metrics.calculate_retrieval_metrics(
    ks=[1, 5, 10, 20, 100],
    use_reordered=False
)
print(results)
```

## After Reranking

```python
from rankify.models.reranking import Reranking

reranker = Reranking(method="monot5", model_name="monot5-base-msmarco")
reranked = reranker.rank(documents)

metrics = Metrics(reranked)
results = metrics.calculate_retrieval_metrics(ks=[1, 5, 10], use_reordered=True)
print(results)
```

## Comparing Retrievers

```python
retrievers = ["bm25", "dpr", "colbert"]
results = {}

for ret in retrievers:
    dataset = Dataset(retriever=ret, dataset_name="nq-dev", n_docs=100)
    docs = dataset.download()
    metrics = Metrics(docs)
    results[ret] = metrics.calculate_retrieval_metrics(ks=[1, 5, 10])

print(results)
```
