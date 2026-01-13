---
title: "Introduction to RAG"
---

# 📌 Introduction to Retrieval-Augmented Generation (RAG)

RAG combines retrieval with language model generation to produce grounded, accurate answers.

## What is RAG?

RAG (Retrieval-Augmented Generation) is a paradigm that:
1. **Retrieves** relevant documents for a query
2. **Augments** the language model's context with retrieved information
3. **Generates** an answer grounded in the retrieved context

## RAG Methods in Rankify

Rankify supports **7 RAG methods**:

| Method | Description |
|--------|-------------|
| Zero-Shot | Direct generation with context |
| Basic RAG | Simple context + question prompting |
| Chain-of-Thought | Step-by-step reasoning |
| Self-Consistency | Multiple reasoning paths |
| ReAct | Reasoning + Action cycles |
| FiD | Fusion-in-Decoder architecture |
| In-Context RALM | In-context retrieval-augmented LM |

## Model Backends

Rankify supports multiple LLM backends:

| Backend | Description | Example Models |
|---------|-------------|----------------|
| huggingface | Local HuggingFace models | LLaMA, Mistral |
| openai | OpenAI API | GPT-4, GPT-3.5 |
| litellm | Multi-provider API | Claude, Gemini |
| vllm | Fast local inference | LLaMA, Mistral |
| fid | Fusion-in-Decoder | FiD-NQ, FiD-TQA |

## Quick Start

```python
from rankify.dataset.dataset import Document, Question, Context
from rankify.generator.generator import Generator

# Create a document with retrieved contexts
question = Question("What is the capital of France?")
contexts = [
    Context(text="Paris is the capital and largest city of France.", id="1"),
    Context(text="France is a country in Western Europe.", id="2"),
]
document = Document(question=question, contexts=contexts)

# Initialize generator
generator = Generator(
    method="basic-rag",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    backend="huggingface"
)

# Generate answer
answers = generator.generate([document])
print(answers[0])  # "Paris"
```

## End-to-End RAG Pipeline

```python
from rankify.dataset.dataset import Document, Question
from rankify.retrievers.retriever import Retriever
from rankify.models.reranking import Reranking
from rankify.generator.generator import Generator

# 1. Create query
documents = [Document(question=Question("Who discovered penicillin?"))]

# 2. Retrieve
retriever = Retriever(method="bm25", n_docs=20, index_type="wiki")
retrieved = retriever.retrieve(documents)

# 3. Rerank
reranker = Reranking(method="monot5", model_name="monot5-base-msmarco")
reranked = reranker.rank(retrieved)

# 4. Generate
generator = Generator(
    method="chain-of-thought-rag",
    model_name="gpt-4o-mini",
    backend="openai"
)
answers = generator.generate(reranked)

print(answers[0])
```

## Choosing a RAG Method

| Use Case | Recommended Method |
|----------|-------------------|
| Simple QA | Basic RAG, Zero-Shot |
| Complex reasoning | Chain-of-Thought |
| High accuracy | Self-Consistency |
| Multi-step tasks | ReAct |
| Encoder-decoder | FiD |

## Next Steps

- [📥 Zero-Shot RAG](zero_shot.md) - Simple generation
- [🔁 Fusion-in-Decoder](fid.md) - FiD architecture
- [📄 In-Context Learning](in_context.md) - Advanced prompting
- [⚙️ Building Pipelines](pipelines.md) - End-to-end systems
