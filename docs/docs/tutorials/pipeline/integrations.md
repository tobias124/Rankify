---
title: "Integrations - LangChain & LlamaIndex"
---

# 🔌 Integrations

Use Rankify with your existing LangChain or LlamaIndex applications.

## LangChain Integration

### Basic Usage

```python
from rankify.integrations import LangChainRetriever
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Create Rankify retriever for LangChain
retriever = LangChainRetriever(
    method="bge",
    reranker="flashrank",
    n_docs=100,
    top_k=10,
)

# Use with LangChain
llm = ChatOpenAI(model="gpt-4o-mini")
chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

result = chain.invoke({"query": "What is machine learning?"})
print(result["result"])
```

**Expected Output:**
```
Machine learning is a subset of artificial intelligence that enables 
systems to learn and improve from experience without being explicitly 
programmed...
```

### With LCEL

```python
from rankify.integrations import LangChainRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

retriever = LangChainRetriever(method="colbert", reranker="monot5")
llm = ChatOpenAI()

prompt = ChatPromptTemplate.from_template("""
Answer the question based on the following context:

{context}

Question: {question}
""")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke("Explain transformers")
print(result)
```

---

## LlamaIndex Integration

### Basic Usage

```python
from rankify.integrations import LlamaIndexRetriever

# Create Rankify retriever for LlamaIndex
retriever = LlamaIndexRetriever(
    method="bge",
    reranker="flashrank",
    top_k=5,
)

# Retrieve nodes
nodes = retriever.retrieve("What is deep learning?")

for node in nodes:
    print(f"Score: {node.score:.3f}")
    print(f"Text: {node.node.text[:100]}...")
```

**Expected Output:**
```
Score: 0.923
Text: Deep learning is a subset of machine learning that uses neural networks...
Score: 0.891
Text: Neural networks in deep learning have multiple hidden layers...
```

### With Query Engine

```python
from rankify.integrations import LlamaIndexRetriever
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI

# Create retriever
retriever = LlamaIndexRetriever(
    method="colbert",
    reranker="monot5",
    top_k=5,
)

# Create query engine
llm = OpenAI(model="gpt-4o-mini")
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    llm=llm,
)

response = query_engine.query("How do transformers work?")
print(response)
```

---

## Configuration Options

### LangChainRetriever

```python
from rankify.integrations import LangChainRetriever

retriever = LangChainRetriever(
    method="bge",          # Retriever method
    reranker="flashrank",  # Optional reranker
    reranker_model=None,   # Specific reranker model
    n_docs=100,            # Docs to retrieve
    top_k=10,              # Docs to return
    index_type="wiki",     # Index type
)
```

### LlamaIndexRetriever

```python
from rankify.integrations import LlamaIndexRetriever

retriever = LlamaIndexRetriever(
    method="colbert",      # Retriever method
    reranker="monot5",     # Optional reranker
    reranker_model=None,   # Specific reranker model
    n_docs=100,            # Docs to retrieve
    top_k=5,               # Docs to return
)
```

---

## Available Methods

### Retrievers
- `bm25`, `dpr`, `bge`, `ance`, `colbert`, `contriever`

### Rerankers
- `flashrank`, `monot5`, `rankgpt`, `inranker`, `colbert_ranker`, `upr`

---

## Why Use Rankify with LangChain/LlamaIndex?

| Feature | LangChain/LlamaIndex | + Rankify |
|---------|---------------------|-----------|
| Retrievers | ~5 built-in | **10+** specialized |
| Rerankers | 1-2 basic | **23+** SOTA methods |
| Hybrid Search | Limited | **Full RRF support** |
| Performance | General purpose | **Optimized for IR** |

---

## Next Steps

- [Web Playground](playground.md) - Interactive UI
- [Pipeline API](index.md) - One-line interface
