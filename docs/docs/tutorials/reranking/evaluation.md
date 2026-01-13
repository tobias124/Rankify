---
title: "Comparing Re-Ranking Performance"
---

# 📈 Comparing Re-Ranking Performance

Learn how to evaluate and compare different reranking methods.

## Evaluation Metrics

Rankify provides these metrics for reranking:

| Metric | Description |
|--------|-------------|
| NDCG@k | Normalized Discounted Cumulative Gain |
| MAP | Mean Average Precision |
| MRR | Mean Reciprocal Rank |
| P@k | Precision at k |
| Recall@k | Recall at k |

## Basic Evaluation

```python
from rankify.dataset.dataset import Dataset
from rankify.models.reranking import Reranking
from rankify.metrics.metrics import Metrics

# Load dataset with ground truth
dataset = Dataset(retriever="bm25", dataset_name="nq-dev", n_docs=100)
documents = dataset.download()

# Rerank with MonoT5
reranker = Reranking(method="monot5", model_name="monot5-base-msmarco")
reranked_docs = reranker.rank(documents)

# Evaluate
metrics = Metrics(reranked_docs)

# Before reranking (original retrieval order)
original_results = metrics.calculate_retrieval_metrics(
    ks=[1, 5, 10, 20],
    use_reordered=False
)
print("Before reranking:", original_results)

# After reranking
reranked_results = metrics.calculate_retrieval_metrics(
    ks=[1, 5, 10, 20],
    use_reordered=True
)
print("After reranking:", reranked_results)
```

## Comparing Multiple Rerankers

```python
from rankify.dataset.dataset import Dataset
from rankify.models.reranking import Reranking
from rankify.metrics.metrics import Metrics
import copy

# Load dataset
dataset = Dataset(retriever="bm25", dataset_name="nq-dev", n_docs=100)
documents = dataset.download()

# Define rerankers to compare
rerankers = {
    "MonoT5-base": Reranking(method="monot5", model_name="monot5-base-msmarco"),
    "RankT5-base": Reranking(method="rankt5", model_name="rankt5-base"),
    "FlashRank": Reranking(method="flashrank", model_name="ms-marco-MiniLM-L-12-v2"),
}

results = {}

for name, reranker in rerankers.items():
    # Deep copy to avoid modifying original
    docs_copy = copy.deepcopy(documents)
    
    # Rerank
    reranked = reranker.rank(docs_copy)
    
    # Evaluate
    metrics = Metrics(reranked)
    results[name] = metrics.calculate_retrieval_metrics(
        ks=[1, 5, 10],
        use_reordered=True
    )
    print(f"{name}: {results[name]}")
```

## Visualization

```python
import matplotlib.pyplot as plt

# Example results
methods = ["BM25", "MonoT5", "RankT5", "FlashRank"]
recall_at_5 = [0.45, 0.62, 0.58, 0.55]
recall_at_10 = [0.55, 0.71, 0.68, 0.64]

x = range(len(methods))
width = 0.35

fig, ax = plt.subplots()
ax.bar([i - width/2 for i in x], recall_at_5, width, label='Recall@5')
ax.bar([i + width/2 for i in x], recall_at_10, width, label='Recall@10')

ax.set_ylabel('Recall')
ax.set_title('Reranker Comparison on NQ-dev')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()

plt.savefig('reranker_comparison.png')
```

## Speed Benchmarking

```python
import time

documents = documents[:100]  # Sample for speed test

speed_results = {}

for name, reranker in rerankers.items():
    docs_copy = copy.deepcopy(documents)
    
    start = time.time()
    reranked = reranker.rank(docs_copy)
    elapsed = time.time() - start
    
    speed_results[name] = {
        "total_time": elapsed,
        "docs_per_second": len(documents) / elapsed
    }
    print(f"{name}: {elapsed:.2f}s ({speed_results[name]['docs_per_second']:.1f} docs/s)")
```

## Creating a Benchmark Report

```python
import pandas as pd

# Combine quality and speed results
report_data = []

for name in rerankers.keys():
    report_data.append({
        "Method": name,
        "Recall@5": results[name].get("Recall@5", 0),
        "Recall@10": results[name].get("Recall@10", 0),
        "MRR": results[name].get("MRR", 0),
        "Speed (docs/s)": speed_results[name]["docs_per_second"]
    })

df = pd.DataFrame(report_data)
print(df.to_markdown(index=False))

# Save to file
df.to_csv("reranker_benchmark.csv", index=False)
```

## Best Practices

1. **Use consistent test sets**: Always compare on the same data
2. **Control for variance**: Average over multiple runs
3. **Consider trade-offs**: Balance quality vs. speed
4. **Test on your domain**: Results may vary by domain

## Next Steps

- [🧠 RAG Introduction](../rag/introduction.md) - Use rerankers in RAG
- [📊 Complete Evaluation](../evaluation/reranking_metrics.md) - Detailed metrics
