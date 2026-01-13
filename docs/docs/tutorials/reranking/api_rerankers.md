---
title: "API-Based Rerankers"
---

# 🦾 API-Based Rerankers (Voyage, Jina, MixedBread.ai)

API-based rerankers provide high-quality reranking without local GPU requirements.


## Supported Providers

| Provider | API | Default Model |
|----------|-----|---------------|
| Cohere | rerank | rerank-english-v3.0 |
| Jina | rerank | jina-reranker-v1-base-en |
| Voyage | rerank | rerank-lite-1 |
| MixedBread.ai | reranking | mxbai-rerank-large-v1 |

## Setup

Get API keys from each provider:
- [Cohere Dashboard](https://dashboard.cohere.com/)
- [Jina AI](https://jina.ai/)
- [Voyage AI](https://www.voyageai.com/)
- [MixedBread.ai](https://www.mixedbread.ai/)

## Cohere Reranker

```python
from rankify.dataset.dataset import Document, Question, Context
from rankify.models.reranking import Reranking

question = Question("What are the benefits of exercise?")
contexts = [
    Context(text="Regular exercise improves cardiovascular health.", id="1"),
    Context(text="Exercise can help reduce stress and anxiety.", id="2"),
    Context(text="The stock market closed higher today.", id="3"),
]
document = Document(question=question, contexts=contexts)

reranker = Reranking(
    method="apiranker",
    model_name="cohere",
    api_key="your-cohere-api-key"
)
reranked = reranker.rank([document])

for ctx in reranked[0].reorder_contexts:
    print(f"[{ctx.score:.4f}] {ctx.text}")
```

## Jina Reranker

```python
reranker = Reranking(
    method="apiranker",
    model_name="jina",
    api_key="your-jina-api-key"
)
reranked = reranker.rank([document])
```

## Voyage Reranker

```python
reranker = Reranking(
    method="apiranker",
    model_name="voyage",
    api_key="your-voyage-api-key"
)
reranked = reranker.rank([document])
```

## MixedBread.ai Reranker

```python
reranker = Reranking(
    method="apiranker",
    model_name="mixedbread.ai",
    api_key="your-mixedbread-api-key"
)
reranked = reranker.rank([document])
```

## Environment Variables

Store API keys securely:

```python
import os

# Set API keys as environment variables
os.environ["COHERE_API_KEY"] = "your-key"
os.environ["JINA_API_KEY"] = "your-key"
os.environ["VOYAGE_API_KEY"] = "your-key"
os.environ["MIXEDBREAD_API_KEY"] = "your-key"

# Use in code
reranker = Reranking(
    method="apiranker",
    model_name="cohere",
    api_key=os.environ["COHERE_API_KEY"]
)
```

## Comparison

| Provider | Speed | Quality | Pricing |
|----------|-------|---------|---------|
| Cohere | Fast | Excellent | Per query |
| Jina | Fast | Very Good | Free tier |
| Voyage | Fast | Very Good | Per query |
| MixedBread | Fast | Excellent | Per query |

## Rate Limiting

Handle rate limits gracefully:

```python
import time
from tqdm import tqdm

documents = [...]  # Many documents

reranker = Reranking(method="apiranker", model_name="cohere", api_key="...")

reranked_docs = []
for doc in tqdm(documents):
    try:
        result = reranker.rank([doc])
        reranked_docs.extend(result)
    except Exception as e:
        print(f"Rate limited, waiting... {e}")
        time.sleep(1)
        result = reranker.rank([doc])
        reranked_docs.extend(result)
```

## Best Practices

1. **Batch requests**: Send multiple documents when possible
2. **Cache results**: Store reranking results to avoid repeated API calls
3. **Use environment variables**: Never hardcode API keys
4. **Handle errors**: Implement retry logic for rate limits

## Next Steps

- [📈 Evaluation](evaluation.md) - Compare API vs local rerankers
- [📊 RAG Pipelines](../rag/pipelines.md) - Build complete systems
