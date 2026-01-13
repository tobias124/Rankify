---
title: "Retrieval Metrics"
---

# 📏 Measuring Retrieval Performance

Evaluate retrieval quality with standard metrics.

## Metrics Overview

| Metric | Description |
|--------|-------------|
| Recall@k | Fraction of relevant docs in top-k |
| P@k | Precision at k |
| MRR | Mean Reciprocal Rank |
| Top-k Accuracy | Whether answer exists in top-k |

## Usage

```python
from rankify.metrics.metrics import Metrics

metrics = Metrics(documents)
results = metrics.calculate_retrieval_metrics(
    ks=[1, 5, 10, 20, 50, 100],
    use_reordered=False  # Original retrieval order
)

print(f"Recall@5: {results['Recall@5']:.4f}")
print(f"Recall@10: {results['Recall@10']:.4f}")
print(f"MRR: {results['MRR']:.4f}")
```

## Interpreting Results

- **Recall@k**: Higher is better, measures coverage
- **MRR**: Measures where first relevant result appears
- **P@k**: Precision among top-k results
