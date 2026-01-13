---
title: "Zero-Shot RAG"
---

# 📥 Zero-Shot RAG (GPT, LLaMA)

Zero-shot RAG provides direct generation from retrieved context without examples.

## Overview

Zero-shot RAG:
- No training or fine-tuning required
- Works with any instruction-tuned LLM
- Simple but effective for straightforward QA

## Using HuggingFace Models

```python
from rankify.dataset.dataset import Document, Question, Context
from rankify.generator.generator import Generator

# Create document
question = Question("When was the Eiffel Tower built?")
contexts = [
    Context(text="The Eiffel Tower was constructed from 1887 to 1889.", id="1"),
    Context(text="Gustave Eiffel designed the famous tower in Paris.", id="2"),
]
document = Document(question=question, contexts=contexts)

# Zero-shot with HuggingFace
generator = Generator(
    method="zero-shot",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    backend="huggingface"
)

answers = generator.generate([document])
print(answers[0])
```

## Using OpenAI API

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

generator = Generator(
    method="zero-shot",
    model_name="gpt-4o-mini",
    backend="openai"
)

answers = generator.generate([document])
print(answers[0])
```

## Using vLLM (Fast Inference)

```python
# Requires: pip install "rankify[reranking]"

generator = Generator(
    method="zero-shot",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    backend="vllm"
)

# Batch generation is much faster with vLLM
documents = [...]  # Multiple documents
answers = generator.generate(documents)
```

## Using LiteLLM (Multi-Provider)

```python
# LiteLLM supports 100+ providers

generator = Generator(
    method="zero-shot",
    model_name="claude-3-5-sonnet-20241022",
    backend="litellm"
)

answers = generator.generate([document])
```

## Custom Prompts

Override the default prompt:

```python
custom_prompt = """Answer the question based only on the provided context.
If the answer is not in the context, say "I don't know."

Context: {context}

Question: {question}

Answer:"""

answers = generator.generate([document], custom_prompt=custom_prompt)
```

## Generation Parameters

Control generation behavior:

```python
answers = generator.generate(
    [document],
    max_new_tokens=256,
    temperature=0.1,  # Lower = more deterministic
    top_p=0.9,
    do_sample=True
)
```

## Basic RAG vs Zero-Shot

For slightly more structured prompting, use basic-rag:

```python
generator = Generator(
    method="basic-rag",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    backend="huggingface"
)
```

## Comparison

| Backend | Speed | Cost | GPU Required |
|---------|-------|------|--------------|
| HuggingFace | Medium | Free | Yes |
| vLLM | Fast | Free | Yes |
| OpenAI | Fast | Paid | No |
| LiteLLM | Fast | Varies | No |

## Next Steps

- [🔁 Fusion-in-Decoder](fid.md) - Specialized RAG architecture
- [📄 In-Context Learning](in_context.md) - Advanced prompting
- [⚙️ Building Pipelines](pipelines.md) - Complete systems
