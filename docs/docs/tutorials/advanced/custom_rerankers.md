---
title: "Custom Rerankers"
---

# 🔧 Implementing a Custom Reranker

Create custom rerankers by extending the base class.

## Creating a Custom Reranker

```python
from rankify.models.base import BaseRanking
from rankify.dataset.dataset import Document
from typing import List

class MyCustomReranker(BaseRanking):
    """Custom reranker implementation."""
    
    def __init__(self, method: str, model_name: str, **kwargs):
        super().__init__(method, model_name, **kwargs)
        # Load your model
        self.model = self._load_model(model_name)
    
    def _load_model(self, model_name: str):
        # Load your reranking model
        pass
    
    def rank(self, documents: List[Document]) -> List[Document]:
        for doc in documents:
            query = doc.question.question
            
            # Score each context
            scored_contexts = []
            for ctx in doc.contexts:
                score = self._score(query, ctx.text)
                ctx.score = score
                scored_contexts.append(ctx)
            
            # Sort by score descending
            doc.reorder_contexts = sorted(
                scored_contexts,
                key=lambda x: x.score,
                reverse=True
            )
        
        return documents
    
    def _score(self, query: str, passage: str) -> float:
        # Implement your scoring logic
        pass
```

## Using Your Custom Reranker

```python
reranker = MyCustomReranker(method="custom", model_name="my-model")
reranked = reranker.rank(documents)
```
