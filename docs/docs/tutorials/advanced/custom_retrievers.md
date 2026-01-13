---
title: "Custom Retrievers"
---

# 🛠 Extending Rankify: Adding Custom Retrievers

Build custom retrievers by extending the base class.

## Creating a Custom Retriever

```python
from rankify.retrievers.base_retriever import BaseRetriever
from rankify.dataset.dataset import Document, Context
from typing import List

class MyCustomRetriever(BaseRetriever):
    """Custom retriever implementation."""
    
    def __init__(self, n_docs: int = 10, **kwargs):
        self.n_docs = n_docs
        # Initialize your search backend
        self.index = self._load_index()
    
    def _load_index(self):
        # Load your custom index
        pass
    
    def retrieve(self, documents: List[Document]) -> List[Document]:
        for doc in documents:
            query = doc.question.question
            
            # Your retrieval logic
            results = self._search(query, self.n_docs)
            
            doc.contexts = [
                Context(
                    text=r["text"],
                    id=r["id"],
                    title=r.get("title", ""),
                    score=r["score"]
                )
                for r in results
            ]
        
        return documents
    
    def _search(self, query: str, k: int):
        # Implement your search logic
        pass
```

## Using Your Custom Retriever

```python
retriever = MyCustomRetriever(n_docs=10)
documents = [Document(question=Question("Test query"))]
results = retriever.retrieve(documents)
```
