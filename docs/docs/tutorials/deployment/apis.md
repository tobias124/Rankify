---
title: "External APIs"
---

# 🌍 Integrating Rankify with External APIs

Use Rankify with external services.

## OpenAI Integration

```python
import os
os.environ["OPENAI_API_KEY"] = "your-key"

from rankify.generator.generator import Generator

generator = Generator(
    method="basic-rag",
    model_name="gpt-4o-mini",
    backend="openai"
)
```

## Cohere Reranking API

```python
reranker = Reranking(
    method="apiranker",
    model_name="cohere",
    api_key=os.environ["COHERE_API_KEY"]
)
```

## LiteLLM (100+ Providers)

```python
generator = Generator(
    method="basic-rag",
    model_name="claude-3-5-sonnet-20241022",
    backend="litellm"
)
```

## Error Handling

```python
import time

def api_call_with_retry(func, max_retries=3):
    for i in range(max_retries):
        try:
            return func()
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(2 ** i)
            else:
                raise e
```
