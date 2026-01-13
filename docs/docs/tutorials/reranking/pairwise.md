---
title: "Pairwise Re-Ranking"
---

# 🔄 Pairwise Re-Ranking (RankGPT, InRanker, EchoRank)

Pairwise rerankers compare document pairs to determine relative ordering.

## RankGPT

RankGPT uses large language models for sophisticated pairwise comparison:

### Using Local Models (vLLM)

```python
from rankify.dataset.dataset import Document, Question, Context
from rankify.models.reranking import Reranking

question = Question("What causes global warming?")
contexts = [
    Context(text="Greenhouse gases trap heat in the atmosphere.", id="1"),
    Context(text="The climate has changed throughout Earth's history.", id="2"),
    Context(text="CO2 emissions from fossil fuels contribute to warming.", id="3"),
]
document = Document(question=question, contexts=contexts)

# Local LLM with vLLM backend
reranker = Reranking(
    method="rankgpt",
    model_name="llamav3.1-8b"
)
reranked = reranker.rank([document])
```

### Using API Models

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

# GPT-4 based ranking
reranker = Reranking(
    method="rankgpt-api",
    model_name="gpt-4",
    api_key=os.environ["OPENAI_API_KEY"]
)
reranked = reranker.rank([document])

# Claude-based ranking
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"
reranker = Reranking(
    method="rankgpt-api",
    model_name="claude-3-5",
    api_key=os.environ["ANTHROPIC_API_KEY"]
)
```

### Available RankGPT Models

| Model | Type | Description |
|-------|------|-------------|
| llamav3.1-8b | Local | Meta LLaMA 3.1 8B |
| llamav3.1-70b | Local | Meta LLaMA 3.1 70B |
| gpt-3.5 | API | OpenAI GPT-3.5 |
| gpt-4 | API | OpenAI GPT-4 |
| claude-3-5 | API | Anthropic Claude |

## InRanker

InRanker uses instruction-tuned models:

```python
reranker = Reranking(method="inranker", model_name="inranker-base")
reranked = reranker.rank([document])
```

### Available InRanker Models

| Model | Size |
|-------|------|
| inranker-small | 60M |
| inranker-base | 220M |
| inranker-3b | 3B |

## EchoRank

EchoRank uses echo-based comparison with T5 models:

```python
reranker = Reranking(method="echorank", model_name="flan-t5-large")
reranked = reranker.rank([document])
```

## In-Context Reranker

Uses in-context learning with LLMs:

```python
reranker = Reranking(
    method="incontext_reranker",
    model_name="llamav3.1-8b"
)
reranked = reranker.rank([document])
```

## Blender Reranker (PairRM)

Uses PairRM for pairwise comparison:

```python
reranker = Reranking(method="blender_reranker", model_name="PairRM")
reranked = reranker.rank([document])
```

## Comparison

| Method | Speed | Quality | GPU Required |
|--------|-------|---------|--------------|
| RankGPT (GPT-4) | Slow | Excellent | No (API) |
| RankGPT (LLaMA) | Medium | Very Good | Yes |
| InRanker | Fast | Good | Optional |
| EchoRank | Medium | Good | Yes |
| PairRM | Medium | Good | Yes |

## Best Practices

1. **Limit context count**: Pairwise comparison is O(n²), keep to <20 contexts
2. **Use API for quality**: GPT-4 provides best quality but higher cost
3. **Local for speed**: Use LLaMA/Mistral for faster local inference

## Next Steps

- [📃 Listwise Reranking](listwise.md) - RankT5, LiT5
- [🦾 API Rerankers](api_rerankers.md) - Cohere, Jina
- [📈 Evaluation](evaluation.md) - Compare methods
