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
from rankify.dataset.dataset import Document, Question, Context

# Create document
question = Question("What is machine learning?")
contexts = [
    Context(text="Machine learning is a subset of AI.", id="1"),
    Context(text="The weather is sunny today.", id="2"),
]
document = Document(question=question, contexts=contexts)

# Use Cohere reranker
reranker = Reranking(
    method="apiranker",
    model_name="cohere",
    api_key="your-cohere-api-key"
)
reranked_docs = reranker.rank([document])
```

## API Reference

::: rankify.models.apiranker
    handler: python
    options:
        show_source: true
        show_undocumented_members: true
        show_root_heading: true
        show_inherited_members: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
