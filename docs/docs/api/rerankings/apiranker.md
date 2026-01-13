# API Reranker

The API Reranker provides integration with external reranking APIs including Cohere, Jina, Voyage, and MixedBread.ai.

## Supported Providers

| Provider | API Key Required | Model |
|----------|------------------|-------|
| Cohere | Yes | rerank-english-v3.0 |
| Jina | Yes | jina-reranker-v1-base-en |
| Voyage | Yes | rerank-lite-1 |
| MixedBread.ai | Yes | mxbai-rerank-large-v1 |

## Usage

```python
from rankify.models.reranking import Reranking

reranker = Reranking(
    method="apiranker",
    model_name="cohere",
    api_key="your-cohere-api-key"
)
reranked_docs = reranker.rank([document])
```

## API Reference

::: rankify.models.apiranker
    options:
        show_source: true
        members: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
