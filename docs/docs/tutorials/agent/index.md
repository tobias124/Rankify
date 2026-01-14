---
title: "RankifyAgent - Intelligent Model Selection"
---

# 🤖 RankifyAgent - Intelligent Model Selection

RankifyAgent is an AI-powered assistant that helps you select the optimal retrieval, reranking, and RAG models for your specific use case.

## Overview

With **10+ retrievers**, **23+ rerankers**, and **7 RAG methods**, choosing the right combination can be overwhelming. RankifyAgent solves this by:

- 📊 **Analyzing your requirements** - Task type, hardware, latency needs
- 🎯 **Recommending optimal models** - Based on accuracy, speed, and constraints
- 💬 **Conversational interface** - Natural language interaction
- 💻 **Code generation** - Ready-to-use code snippets

## Installation

RankifyAgent is included with Rankify:

```bash
pip install rankify
```

For conversational features, install OpenAI:

```bash
pip install openai litellm
```

---

## Quick Start

### 1. Programmatic Recommendation

Get instant recommendations with a single function call:

```python
from rankify.agent import recommend

# Recommend models for QA task with GPU
result = recommend(task="qa", gpu=True, include_rag=True)

print(f"Retriever: {result.retriever.name}")
print(f"Reranker: {result.reranker.name}")
print(f"RAG Method: {result.rag_method.name}")
```

**Expected Output:**
```
Retriever: BGE
Reranker: MonoT5 Base
RAG Method: Chain-of-Thought RAG
```

### 2. Conversational Agent

Have a natural conversation to get recommendations:

```python
from rankify.agent import RankifyAgent
import os

# Set up Azure OpenAI (or use other backends)
os.environ["AZURE_OPENAI_ENDPOINT"] = "your-endpoint"
os.environ["AZURE_OPENAI_API_KEY"] = "your-key"
os.environ["AZURE_DEPLOYMENT_NAME"] = "gpt-4o"

agent = RankifyAgent(backend="azure")
response = agent.chat("I need a fast search system for production without GPU")

print(response.message)
```

**Expected Output:**
```
For a production search system without GPU, I recommend:

**Retriever: BM25**
- Ultra-fast sparse retrieval, no GPU needed
- Great for keyword-based search

**Reranker: FlashRank MiniLM**
- ONNX-based, runs on CPU
- Very low latency (~10ms)

This combination gives you sub-50ms latency with good accuracy.
```

---

## Supported LLM Backends

RankifyAgent supports multiple LLM backends:

| Backend | Description | Setup |
|---------|-------------|-------|
| `azure` | Azure OpenAI | Set `AZURE_OPENAI_*` env vars |
| `openai` | OpenAI API | Set `OPENAI_API_KEY` |
| `litellm` | 100+ providers | Model-specific setup |
| `local` | Local LLMs | Requires GPU, uses transformers |

### Azure OpenAI Setup

```python
import os

os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "your-api-key"
os.environ["AZURE_DEPLOYMENT_NAME"] = "gpt-4o"
os.environ["AZURE_API_VERSION"] = "2024-05-01-preview"

from rankify.agent import RankifyAgent
agent = RankifyAgent(backend="azure")
```

### OpenAI Setup

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

from rankify.agent import RankifyAgent
agent = RankifyAgent(backend="openai", model_name="gpt-4o-mini")
```

### LiteLLM (Claude, Gemini, etc.)

```python
from rankify.agent import RankifyAgent

# Use Claude
agent = RankifyAgent(backend="litellm", model_name="claude-3-5-sonnet-20241022")

# Use Gemini
agent = RankifyAgent(backend="litellm", model_name="gemini/gemini-pro")
```

---

## API Reference

### `recommend()` Function

Quick recommendation without conversation.

```python
from rankify.agent import recommend

result = recommend(
    task="qa",           # "qa", "search", "summarization", "conversational"
    gpu=True,            # GPU availability
    api_allowed=True,    # Allow API-based models
    include_rag=False,   # Include RAG method recommendation
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | str | "qa" | Task type |
| `gpu` | bool | True | GPU available |
| `api_allowed` | bool | True | Allow API models |
| `include_rag` | bool | False | Include RAG recommendation |

**Returns:** `RecommendationResult`

---

### `RankifyRecommender` Class

More control over recommendations.

```python
from rankify.agent import RankifyRecommender

recommender = RankifyRecommender()

# Full recommendation with constraints
result = recommender.recommend(
    task="search",
    constraints={
        "gpu": False,
        "prefer_speed": True,
        "no_api": True,
    },
    include_reranker=True,
    include_rag=False,
    top_k=3,  # Number of alternatives
)

print(f"Best: {result.retriever.name}")
print(f"Alternatives: {[m.name for m in result.alternatives['retrievers']]}")
```

**Expected Output:**
```
Best: BM25
Alternatives: ['Contriever', 'DPR (Single-Encoder)']
```

#### Specialized Methods

```python
# Optimize for latency
result = recommender.recommend_for_latency(max_latency_ms=50, task="search")

# Optimize for accuracy
result = recommender.recommend_for_accuracy(task="qa", gpu_available=True)

# Production-ready models
result = recommender.recommend_for_production(gpu_available=False, api_allowed=True)
```

---

### `RankifyAgent` Class

Conversational interface with memory.

```python
from rankify.agent import RankifyAgent

agent = RankifyAgent(backend="azure")

# First message
response = agent.chat("I need to build a QA system for legal documents")
print(response.message)

# Follow-up (remembers context)
response = agent.chat("What if I also need summarization?")
print(response.message)

# Get code snippet
if response.code_snippet:
    print(response.code_snippet)

# Clear conversation history
agent.clear_history()
```

**Response Object:**

```python
@dataclass
class AgentResponse:
    message: str                              # Agent's response
    recommendation: RecommendationResult      # Recommendation object
    code_snippet: str                         # Generated code
```

---

## Constraint Options

| Constraint | Type | Description |
|------------|------|-------------|
| `gpu` | bool | GPU availability |
| `max_memory_mb` | int | Maximum memory budget |
| `prefer_speed` | bool | Prioritize fast models |
| `prefer_accuracy` | bool | Prioritize accurate models |
| `no_api` | bool | Only local models |
| `api_only` | bool | Only API-based models |
| `language` | str | Required language ("en", "zh", etc.) |

---

## Model Registry

Explore available models:

```python
from rankify.agent import (
    get_all_retrievers,
    get_all_rerankers, 
    get_all_rag_methods,
)

# List all retrievers
for name, model in get_all_retrievers().items():
    print(f"{model.name}: {model.speed.value}, GPU={model.gpu_required}")
```

**Expected Output:**
```
BM25: very_fast, GPU=False
DPR (Multi-Encoder): medium, GPU=True
ANCE: medium, GPU=True
BGE: fast, GPU=True
ColBERT: slow, GPU=True
Contriever: medium, GPU=True
HyDE: slow, GPU=True
Online Retriever: medium, GPU=False
```

### Model Metadata

Each model has detailed metadata:

```python
from rankify.agent import get_model

model = get_model("reranker", "flashrank-minilm")
print(f"Name: {model.name}")
print(f"Speed: {model.speed.value}")
print(f"Accuracy: {model.accuracy.value}")
print(f"GPU Required: {model.gpu_required}")
print(f"Memory: {model.memory_mb}MB")
print(f"Best For: {model.best_for}")
```

**Expected Output:**
```
Name: FlashRank MiniLM
Speed: very_fast
Accuracy: good
GPU Required: False
Memory: 50MB
Best For: ['low latency', 'cpu deployment', 'edge devices', 'production']
```

---

## Complete Example

End-to-end usage from recommendation to execution:

```python
from rankify.agent import recommend
from rankify.dataset.dataset import Dataset
from rankify.retrievers.retriever import Retriever
from rankify.models.reranking import Reranking

# Step 1: Get recommendation
result = recommend(task="qa", gpu=True)
print(f"Using: {result.retriever.name} + {result.reranker.name}")

# Step 2: Load data
dataset = Dataset(retriever="bm25", dataset_name="nq-dev", n_docs=100)
documents = dataset.download()[:10]

# Step 3: Use recommended retriever
retriever = Retriever(
    method=result.retriever.method,
    n_docs=100
)
documents = retriever.retrieve(documents)

# Step 4: Use recommended reranker
reranker = Reranking(
    method=result.reranker.method,
    model_name=result.reranker.model_path
)
documents = reranker.rank(documents)

print(f"Processed {len(documents)} documents")
```

---

## Next Steps

- [Retrieval Tutorials](../retrieval/introduction.md) - Learn about retrieval methods
- [Reranking Tutorials](../reranking/introduction.md) - Explore reranking options
- [RAG Tutorials](../rag/introduction.md) - Build RAG pipelines
