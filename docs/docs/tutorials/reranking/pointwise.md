---
title: "Pointwise Re-Ranking"
---

# 🎯 Pointwise Re-Ranking (MonoBERT, MonoT5)

Pointwise rerankers score each query-document pair independently, then sort by score.

## MonoT5


MonoT5 uses T5 to predict relevance ("true" or "false"):

```python
from rankify.dataset.dataset import Document, Question, Context
from rankify.models.reranking import Reranking

# Create document
question = Question("What is the speed of light?")
contexts = [
    Context(text="The speed of light in vacuum is 299,792,458 m/s.", id="1"),
    Context(text="Light travels faster than sound.", id="2"),
    Context(text="Einstein developed the theory of relativity.", id="3"),
]
document = Document(question=question, contexts=contexts)

# MonoT5 reranking
reranker = Reranking(method="monot5", model_name="monot5-base-msmarco")
reranked = reranker.rank([document])

# Results
for ctx in reranked[0].reorder_contexts:
    print(f"[{ctx.score:.4f}] {ctx.text[:80]}...")
```

### Available MonoT5 Models

| Model Name | Size | HuggingFace ID |
|------------|------|----------------|
| monot5-base-msmarco | 220M | castorini/monot5-base-msmarco |
| monot5-large-msmarco | 770M | castorini/monot5-large-msmarco |
| monot5-3b-msmarco-10k | 3B | castorini/monot5-3b-msmarco-10k |

```python
# Using a larger model
reranker = Reranking(method="monot5", model_name="monot5-large-msmarco")
```

## MonoBERT

MonoBERT uses BERT cross-encoders:

```python
reranker = Reranking(method="monobert", model_name="monobert-large")
reranked = reranker.rank([document])
```

## UPR (Unsupervised Passage Reranker)

UPR uses language models without task-specific training:

```python
# T5-based UPR
reranker = Reranking(method="upr", model_name="t5-base")
reranked = reranker.rank([document])

# GPT-2 based UPR
reranker = Reranking(method="upr", model_name="gpt2")
reranked = reranker.rank([document])
```

### Available UPR Models

| Model | Description |
|-------|-------------|
| t5-small | Google T5 Small |
| t5-base | Google T5 Base |
| t5-large | Google T5 Large |
| gpt2 | OpenAI GPT-2 |
| gpt-neo-2.7b | EleutherAI GPT-Neo |
| flan-t5-xl | Google Flan-T5 XL |

## FlashRank (Fast ONNX Reranking)

FlashRank uses ONNX for fast CPU inference:

```python
# Fast, lightweight reranking
reranker = Reranking(method="flashrank", model_name="ms-marco-TinyBERT-L-2-v2")
reranked = reranker.rank([document])
```

### FlashRank Models

| Model | Speed | Quality |
|-------|-------|---------|
| ms-marco-TinyBERT-L-2-v2 | ⚡⚡⚡ | Good |
| ms-marco-MiniLM-L-12-v2 | ⚡⚡ | Very Good |
| rank-T5-flan | ⚡ | Excellent |

## Batch Processing

Process multiple documents efficiently:

```python
# Create multiple documents
documents = [
    Document(question=Question("Who invented the telephone?"), contexts=[...]),
    Document(question=Question("What is DNA?"), contexts=[...]),
    Document(question=Question("When was the moon landing?"), contexts=[...]),
]

# Batch reranking
reranker = Reranking(method="monot5", model_name="monot5-base-msmarco")
reranked_docs = reranker.rank(documents)

for doc in reranked_docs:
    print(f"Q: {doc.question.question}")
    print(f"Top result: {doc.reorder_contexts[0].text[:100]}...")
```

## Next Steps

- [🔄 Pairwise Reranking](pairwise.md) - RankGPT, InRanker
- [📃 Listwise Reranking](listwise.md) - RankT5, LiT5
- [📈 Evaluation](evaluation.md) - Compare reranker performance
