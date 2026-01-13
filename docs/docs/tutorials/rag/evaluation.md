---
title: "Evaluating RAG Models"
---

# 📊 Evaluating RAG Models

Learn to measure RAG performance with Rankify's evaluation metrics.

## Evaluation Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| Exact Match (EM) | Exact string match | Factoid QA |
| F1 Score | Token-level overlap | Short answers |
| Contains Match | Answer in prediction | Long-form answers |
| BLEU | N-gram precision | Text generation |
| ROUGE | N-gram recall | Summarization |

## Basic Evaluation

```python
from rankify.dataset.dataset import Dataset
from rankify.generator.generator import Generator
from rankify.metrics.metrics import Metrics

# Load dataset with ground truth answers
dataset = Dataset(retriever="bm25", dataset_name="nq-dev", n_docs=100)
documents = dataset.download()[:100]

# Generate answers
generator = Generator(
    method="basic-rag",
    model_name="gpt-4o-mini",
    backend="openai"
)
predictions = generator.generate(documents)

# Evaluate
metrics = Metrics(documents)
results = metrics.calculate_generation_metrics(predictions)

print(f"Exact Match: {results['exact_match']:.4f}")
print(f"F1 Score: {results['f1_score']:.4f}")
print(f"Contains Match: {results['contains_match']:.4f}")
```

## Per-Question Analysis

```python
# Get individual scores
metrics = Metrics(documents)
summary, individual_scores = metrics.calculate_generation_metrics(
    predictions,
    return_individual=True
)

# Find incorrect predictions
for i, (doc, pred, score) in enumerate(zip(documents, predictions, individual_scores)):
    if score['exact_match'] == 0:
        print(f"Q: {doc.question.question}")
        print(f"Gold: {doc.answers.answers}")
        print(f"Pred: {pred}")
        print("---")
```

## Comparing RAG Methods

```python
import copy

# Methods to compare
methods = {
    "zero-shot": Generator(method="zero-shot", ...),
    "basic-rag": Generator(method="basic-rag", ...),
    "chain-of-thought": Generator(method="chain-of-thought-rag", ...),
}

results = {}

for name, generator in methods.items():
    docs_copy = copy.deepcopy(documents)
    predictions = generator.generate(docs_copy)
    
    metrics = Metrics(docs_copy)
    results[name] = metrics.calculate_generation_metrics(predictions)
    
    print(f"{name}: EM={results[name]['exact_match']:.3f}, F1={results[name]['f1_score']:.3f}")
```

## End-to-End Evaluation

Evaluate the complete pipeline:

```python
from rankify.retrievers.retriever import Retriever
from rankify.models.reranking import Reranking

# Fresh documents without contexts
documents = [Document(question=Question(d.question.question), 
                     answers=d.answers) 
             for d in original_documents]

# Retrieve
retriever = Retriever(method="bm25", n_docs=50, index_type="wiki")
retrieved = retriever.retrieve(documents)

# Evaluate retrieval
metrics = Metrics(retrieved)
retrieval_results = metrics.calculate_retrieval_metrics(
    ks=[1, 5, 10, 20],
    use_reordered=False
)
print("Retrieval:", retrieval_results)

# Rerank
reranker = Reranking(method="monot5", model_name="monot5-base-msmarco")
reranked = reranker.rank(retrieved)

# Evaluate reranking
rerank_results = metrics.calculate_retrieval_metrics(
    ks=[1, 5, 10, 20],
    use_reordered=True
)
print("After Reranking:", rerank_results)

# Generate
generator = Generator(method="basic-rag", ...)
predictions = generator.generate(reranked)

# Evaluate generation
gen_results = metrics.calculate_generation_metrics(predictions)
print("Generation:", gen_results)
```

## Creating an Evaluation Report

```python
import pandas as pd

def create_evaluation_report(documents, predictions, output_path="report.csv"):
    """Create detailed evaluation report."""
    
    metrics = Metrics(documents)
    summary, individual = metrics.calculate_generation_metrics(
        predictions, return_individual=True
    )
    
    # Build report
    report_data = []
    for doc, pred, scores in zip(documents, predictions, individual):
        report_data.append({
            "question": doc.question.question,
            "gold_answer": str(doc.answers.answers),
            "prediction": pred,
            "exact_match": scores['exact_match'],
            "f1_score": scores['f1_score'],
            "contains_match": scores['contains_match'],
        })
    
    df = pd.DataFrame(report_data)
    df.to_csv(output_path, index=False)
    
    print(f"Summary: EM={summary['exact_match']:.3f}, F1={summary['f1_score']:.3f}")
    return df
```

## Best Practices

1. **Use consistent test sets**: Compare on the same data
2. **Report multiple metrics**: EM alone doesn't capture partial matches
3. **Analyze errors**: Look at failure cases systematically
4. **Statistical significance**: Average over multiple runs

## Next Steps

- [📏 Retrieval Metrics](../evaluation/retrieval_metrics.md) - Detailed retrieval evaluation
- [📈 Reranking Metrics](../evaluation/reranking_metrics.md) - Detailed reranking evaluation
