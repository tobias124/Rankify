---
title: "RAG Metrics"
---

# 🧠 Evaluating RAG Outputs

Measure RAG generation quality.

## Key Metrics

| Metric | Description |
|--------|-------------|
| Exact Match | Exact string match |
| F1 Score | Token-level F1 |
| Contains Match | Answer in prediction |

## Usage

```python
from rankify.generator.generator import Generator
from rankify.metrics.metrics import Metrics

generator = Generator(method="basic-rag", model_name="gpt-4o-mini", backend="openai")
predictions = generator.generate(documents)

metrics = Metrics(documents)
results = metrics.calculate_generation_metrics(predictions)

print(f"Exact Match: {results['exact_match']:.4f}")
print(f"F1 Score: {results['f1_score']:.4f}")
print(f"Contains Match: {results['contains_match']:.4f}")
```

## Per-Example Analysis

```python
summary, individual = metrics.calculate_generation_metrics(predictions, return_individual=True)

for doc, pred, scores in zip(documents[:5], predictions[:5], individual[:5]):
    print(f"Q: {doc.question.question}")
    print(f"Pred: {pred}")
    print(f"EM: {scores['exact_match']}, F1: {scores['f1_score']:.3f}")
    print("---")
```
