---
title: "Listwise Re-Ranking"
---

# 📃 Listwise Re-Ranking (RankT5, LiT5, Transformer Rankers)

Listwise rerankers consider the entire document list simultaneously.

## RankT5

RankT5 generates rankings for document lists:

```python
from rankify.dataset.dataset import Document, Question, Context
from rankify.models.reranking import Reranking

question = Question("What is machine learning?")
contexts = [
    Context(text="Machine learning is a subset of artificial intelligence.", id="1"),
    Context(text="Deep learning uses neural networks with many layers.", id="2"),
    Context(text="The weather forecast predicts rain tomorrow.", id="3"),
]
document = Document(question=question, contexts=contexts)

reranker = Reranking(method="rankt5", model_name="rankt5-base")
reranked = reranker.rank([document])
```

### Available RankT5 Models

| Model | Size |
|-------|------|
| rankt5-base | 220M |
| rankt5-large | 770M |
| rankt5-3b | 3B |

## ListT5

ListT5 is optimized for listwise ranking:

```python
reranker = Reranking(method="listt5", model_name="listt5-base")
reranked = reranker.rank([document])
```

### Available ListT5 Models

| Model | Size |
|-------|------|
| listt5-base | 220M |
| listt5-3b | 3B |

## LiT5 (Lightweight T5)

LiT5 offers fast listwise reranking:

```python
# Score-based LiT5
reranker = Reranking(method="lit5score", model_name="LiT5-Score-base")
reranked = reranker.rank([document])

# Distilled LiT5
reranker = Reranking(method="lit5dist", model_name="LiT5-Distill-base")
reranked = reranker.rank([document])
```

!!! note "vLLM Required"
    LiT5 models require vLLM: `pip install "rankify[reranking]"`

## Transformer Reranker (Cross-Encoders)

Cross-encoder models for high-quality reranking:

```python
# MixedBread rerankers
reranker = Reranking(
    method="transformer_ranker",
    model_name="mxbai-rerank-large"
)
reranked = reranker.rank([document])

# BGE rerankers
reranker = Reranking(
    method="transformer_ranker",
    model_name="bge-reranker-large"
)

# Jina rerankers
reranker = Reranking(
    method="transformer_ranker",
    model_name="jina-reranker-base-multilingual"
)
```

### Available Transformer Models

| Model | Language | Quality |
|-------|----------|---------|
| mxbai-rerank-xsmall | English | Good |
| mxbai-rerank-base | English | Very Good |
| mxbai-rerank-large | English | Excellent |
| bge-reranker-base | English | Very Good |
| bge-reranker-large | English | Excellent |
| bge-reranker-v2-m3 | Multilingual | Excellent |
| jina-reranker-v1-tiny-en | English | Good |
| jina-reranker-v2-base-multilingual | Multilingual | Very Good |

## ColBERT Reranker

Late interaction reranking with ColBERT:

```python
reranker = Reranking(method="colbert_ranker", model_name="colbertv2.0")
reranked = reranker.rank([document])
```

### ColBERT Variants

| Model | Language |
|-------|----------|
| colbertv2.0 | English |
| FranchColBERT | French |
| JapanColBERT | Japanese |
| SpanishColBERT | Spanish |
| ArabicColBERT-250k | Arabic |

## TwoLAR

Two-stage listwise reranker:

```python
reranker = Reranking(method="twolar", model_name="twolar-large")
reranked = reranker.rank([document])
```

## Comparison

| Method | Speed | Quality | List Size |
|--------|-------|---------|-----------|
| RankT5 | Medium | Very Good | Up to 100 |
| ListT5 | Medium | Very Good | Up to 100 |
| LiT5 | Fast | Good | Up to 100 |
| Transformer | Fast | Excellent | Unlimited |
| ColBERT | Medium | Excellent | Unlimited |

## Next Steps

- [🦾 API Rerankers](api_rerankers.md) - External APIs
- [📈 Evaluation](evaluation.md) - Benchmarking
