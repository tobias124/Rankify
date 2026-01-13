---
title: "Building RAG Pipelines"
---

# ⚙️ Building RAG Pipelines

Learn to build complete end-to-end RAG systems with Rankify.

## Basic Pipeline

A standard RAG pipeline: Retrieve → Rerank → Generate

```python
from rankify.dataset.dataset import Document, Question
from rankify.retrievers.retriever import Retriever
from rankify.models.reranking import Reranking
from rankify.generator.generator import Generator

def rag_pipeline(query: str) -> str:
    """Complete RAG pipeline."""
    
    # 1. Create document from query
    document = Document(question=Question(query))
    
    # 2. Retrieve
    retriever = Retriever(method="bm25", n_docs=50, index_type="wiki")
    retrieved = retriever.retrieve([document])
    
    # 3. Rerank
    reranker = Reranking(method="monot5", model_name="monot5-base-msmarco")
    reranked = reranker.rank(retrieved)
    
    # 4. Generate
    generator = Generator(
        method="basic-rag",
        model_name="gpt-4o-mini",
        backend="openai"
    )
    answers = generator.generate(reranked)
    
    return answers[0]

# Usage
answer = rag_pipeline("Who invented the light bulb?")
print(answer)
```

## Hybrid Retrieval Pipeline

Combine BM25 and dense retrieval:

```python
def hybrid_rag_pipeline(query: str) -> str:
    """Hybrid retrieval RAG pipeline."""
    
    document = Document(question=Question(query))
    
    # BM25 retrieval
    bm25 = Retriever(method="bm25", n_docs=30, index_type="wiki")
    bm25_results = bm25.retrieve([document])
    
    # Dense retrieval
    dense = Retriever(method="contriever", n_docs=30, index_type="wiki")
    dense_results = dense.retrieve([Document(question=Question(query))])
    
    # Merge results (RRF)
    merged_contexts = merge_results(bm25_results[0], dense_results[0])
    document.contexts = merged_contexts[:50]
    
    # Rerank merged results
    reranker = Reranking(method="monot5", model_name="monot5-base-msmarco")
    reranked = reranker.rank([document])
    
    # Generate
    generator = Generator(method="chain-of-thought-rag", ...)
    return generator.generate(reranked)[0]

def merge_results(doc1, doc2, k=60):
    """Reciprocal Rank Fusion."""
    scores = {}
    for rank, ctx in enumerate(doc1.contexts):
        scores[ctx.id] = scores.get(ctx.id, 0) + 1/(k + rank)
    for rank, ctx in enumerate(doc2.contexts):
        scores[ctx.id] = scores.get(ctx.id, 0) + 1/(k + rank)
    
    all_contexts = {ctx.id: ctx for ctx in doc1.contexts + doc2.contexts}
    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return [all_contexts[id] for id in sorted_ids]
```

## Configurable Pipeline Class

```python
class RAGPipeline:
    """Configurable RAG pipeline."""
    
    def __init__(
        self,
        retriever_method: str = "bm25",
        reranker_method: str = "monot5",
        generator_method: str = "basic-rag",
        n_retrieve: int = 50,
        n_rerank: int = 10,
        model_name: str = "gpt-4o-mini",
        backend: str = "openai"
    ):
        self.retriever = Retriever(
            method=retriever_method,
            n_docs=n_retrieve,
            index_type="wiki"
        )
        self.reranker = Reranking(
            method=reranker_method,
            model_name="monot5-base-msmarco"
        )
        self.generator = Generator(
            method=generator_method,
            model_name=model_name,
            backend=backend
        )
        self.n_rerank = n_rerank
    
    def __call__(self, query: str) -> str:
        document = Document(question=Question(query))
        
        # Retrieve
        retrieved = self.retriever.retrieve([document])
        
        # Rerank and truncate
        reranked = self.reranker.rank(retrieved)
        reranked[0].contexts = reranked[0].reorder_contexts[:self.n_rerank]
        
        # Generate
        answers = self.generator.generate(reranked)
        return answers[0]

# Usage
pipeline = RAGPipeline(
    retriever_method="contriever",
    reranker_method="flashrank",
    generator_method="chain-of-thought-rag",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    backend="huggingface"
)

answer = pipeline("Explain quantum entanglement")
```

## Batch Processing

Process multiple queries efficiently:

```python
def batch_rag(queries: list[str]) -> list[str]:
    """Batch RAG processing."""
    
    # Create documents
    documents = [Document(question=Question(q)) for q in queries]
    
    # Batch retrieve
    retriever = Retriever(method="bm25", n_docs=50, index_type="wiki")
    retrieved = retriever.retrieve(documents)
    
    # Batch rerank
    reranker = Reranking(method="monot5", model_name="monot5-base-msmarco")
    reranked = reranker.rank(retrieved)
    
    # Batch generate
    generator = Generator(method="basic-rag", ...)
    answers = generator.generate(reranked)
    
    return answers
```

## Evaluation Pipeline

```python
from rankify.metrics.metrics import Metrics

def evaluate_pipeline(documents: list, pipeline: RAGPipeline):
    """Evaluate RAG pipeline."""
    
    # Generate answers
    answers = []
    for doc in documents:
        answer = pipeline(doc.question.question)
        answers.append(answer)
    
    # Calculate metrics
    metrics = Metrics(documents)
    results = metrics.calculate_generation_metrics(answers)
    
    return results
```

## Next Steps

- [📊 RAG Evaluation](evaluation.md) - Measure performance
- [🛠 Custom RAG](../advanced/custom_rag.md) - Build custom methods
