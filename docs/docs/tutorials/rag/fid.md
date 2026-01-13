---
title: "Fusion-in-Decoder"
---

# 🔁 Fusion-in-Decoder (FiD)

FiD (Fusion-in-Decoder) is a specialized architecture that processes multiple passages independently then fuses them in the decoder.

## Overview

FiD architecture:
- Encodes each passage separately with the question
- Concatenates all encoded representations
- Decoder attends to all passages simultaneously
- Optimized for knowledge-intensive tasks

## Available FiD Models

| Model | Dataset | Size |
|-------|---------|------|
| nq_reader_base | Natural Questions | 220M |
| nq_reader_large | Natural Questions | 770M |
| tqa_reader_base | TriviaQA | 220M |
| tqa_reader_large | TriviaQA | 770M |

## Basic Usage

```python
from rankify.dataset.dataset import Document, Question, Context
from rankify.generator.generator import Generator

# Create document with multiple contexts
question = Question("What is the largest planet in our solar system?")
contexts = [
    Context(text="Jupiter is the largest planet in the Solar System.", id="1"),
    Context(text="Saturn is the second-largest planet after Jupiter.", id="2"),
    Context(text="The Great Red Spot is a storm on Jupiter.", id="3"),
]
document = Document(question=question, contexts=contexts)

# Initialize FiD generator
generator = Generator(
    method="fid",
    model_name="nq_reader_base",
    backend="fid"
)

# Generate answer
answers = generator.generate([document])
print(answers[0])  # "Jupiter"
```

## Using FiD with Retrieved Documents

```python
from rankify.dataset.dataset import Dataset
from rankify.generator.generator import Generator

# Load pre-retrieved NQ dataset
dataset = Dataset(retriever="bm25", dataset_name="nq-dev", n_docs=100)
documents = dataset.download()[:100]  # First 100 examples

# Initialize FiD
generator = Generator(
    method="fid",
    model_name="nq_reader_base",
    backend="fid"
)

# Generate answers
answers = generator.generate(documents)

for doc, answer in zip(documents[:5], answers[:5]):
    print(f"Q: {doc.question.question}")
    print(f"A: {answer}")
    print("---")
```

## FiD Configuration

Control number of passages used:

```python
# Use top-10 contexts for generation
generator = Generator(
    method="fid",
    model_name="nq_reader_base",
    backend="fid",
    n_contexts=10
)
```

## FiD vs Standard RAG

| Aspect | FiD | Standard RAG |
|--------|-----|--------------|
| Architecture | T5-based encoder-decoder | Decoder-only LLM |
| Passage handling | Independent encoding, joint decoding | Concatenated in prompt |
| Context limit | Flexible (independent encoding) | Limited by context window |
| Training | Task-specific fine-tuning | Pre-trained, prompt-based |
| Speed | Fast | Varies |

## When to Use FiD

✅ **Use FiD when:**
- You have many passages to process
- Extractive QA tasks
- Speed is important
- Task-specific fine-tuning is acceptable

❌ **Avoid FiD when:**
- You need general reasoning
- You want to use the latest LLMs
- You need generation flexibility

## Combining with Reranking

```python
from rankify.models.reranking import Reranking

# Retrieve
retriever = Retriever(method="bm25", n_docs=100, index_type="wiki")
retrieved = retriever.retrieve(documents)

# Rerank to get best passages
reranker = Reranking(method="monot5", model_name="monot5-base-msmarco")
reranked = reranker.rank(retrieved)

# FiD on top-reranked passages
generator = Generator(method="fid", model_name="nq_reader_large", backend="fid")
answers = generator.generate(reranked)
```

## Next Steps

- [📄 In-Context Learning](in_context.md) - Alternative approach
- [⚙️ Building Pipelines](pipelines.md) - Complete systems
- [📊 Evaluation](evaluation.md) - Measure performance
