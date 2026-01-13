---
title: "Custom RAG Models"
---

# ⚙️ Creating Custom RAG Models

Build custom RAG methods by extending the base class.

## Creating a Custom RAG Method

```python
from rankify.generator.rag_methods.base_rag_method import BaseRAGMethod
from rankify.dataset.dataset import Document
from typing import List

class MyCustomRAG(BaseRAGMethod):
    """Custom RAG method."""
    
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
    
    def answer_questions(
        self,
        documents: List[Document],
        custom_prompt: str = None,
        **kwargs
    ) -> List[str]:
        answers = []
        
        for doc in documents:
            # Build context from top contexts
            context = self._build_context(doc)
            
            # Create prompt
            prompt = self._create_prompt(doc.question.question, context)
            
            # Generate answer
            answer = self.model.generate(prompt, **kwargs)
            answers.append(answer)
        
        return answers
    
    def _build_context(self, doc: Document) -> str:
        contexts = doc.reorder_contexts or doc.contexts
        return "\n".join([ctx.text for ctx in contexts[:5]])
    
    def _create_prompt(self, question: str, context: str) -> str:
        return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
```

## Using Your Custom RAG

```python
from rankify.generator.models.model_factory import model_factory

model = model_factory(model_name="gpt-4o-mini", backend="openai")
rag = MyCustomRAG(model)
answers = rag.answer_questions(documents)
```
