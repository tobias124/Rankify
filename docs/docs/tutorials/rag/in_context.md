---
title: "In-Context Learning for RAG"
---

# 📄 In-Context Learning for RAG

In-context learning uses carefully crafted prompts to guide RAG generation.

## Overview

In-context RAG methods in Rankify:
- **Chain-of-Thought RAG**: Step-by-step reasoning
- **Self-Consistency RAG**: Multiple reasoning paths
- **ReAct RAG**: Reasoning + Action cycles
- **In-Context RALM**: Retrieval-augmented language modeling

## Chain-of-Thought RAG

Enables step-by-step reasoning:

```python
from rankify.dataset.dataset import Document, Question, Context
from rankify.generator.generator import Generator

question = Question("What is 15% of 200, then add 50?")
contexts = [
    Context(text="Percentages are calculated by multiplying by the decimal form.", id="1"),
]
document = Document(question=question, contexts=contexts)

generator = Generator(
    method="chain-of-thought-rag",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    backend="huggingface"
)

answers = generator.generate([document])
# Output includes reasoning steps
print(answers[0])
```

## Self-Consistency RAG

Generates multiple answers and selects the most consistent:

```python
generator = Generator(
    method="self-consistency-rag",
    model_name="gpt-4o-mini",
    backend="openai",
    num_samples=5,  # Generate 5 answers
    temperature=0.7  # Higher temperature for diversity
)

answers = generator.generate([document])
# Returns the most consistent answer
print(answers[0])
```

## ReAct RAG

Combines reasoning with actions (e.g., search):

```python
generator = Generator(
    method="react-rag",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    backend="huggingface"
)

# ReAct can use tools for multi-step reasoning
answers = generator.generate([document])
```

## In-Context RALM

Retrieval-augmented language modeling:

```python
generator = Generator(
    method="in-context-ralm",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    backend="huggingface"
)

answers = generator.generate([document])
```

## Custom Prompts

Create custom in-context prompts:

```python
# Chain-of-thought prompt
cot_prompt = """You are given a question and relevant context.
Think step by step to answer the question.

Context:
{context}

Question: {question}

Let's think step by step:
1."""

generator = Generator(
    method="basic-rag",
    model_name="gpt-4o-mini",
    backend="openai"
)

answers = generator.generate([document], custom_prompt=cot_prompt)
```

## Few-Shot In-Context Learning

Add examples to improve performance:

```python
few_shot_prompt = """Answer questions based on the provided context.

Example 1:
Context: The Eiffel Tower was built in 1889 in Paris.
Question: When was the Eiffel Tower built?
Answer: 1889

Example 2:
Context: Albert Einstein developed the theory of relativity.
Question: What did Einstein develop?
Answer: The theory of relativity

Now answer this question:
Context: {context}
Question: {question}
Answer:"""

answers = generator.generate([document], custom_prompt=few_shot_prompt)
```

## Method Comparison

| Method | Reasoning | Consistency | Speed |
|--------|-----------|-------------|-------|
| Zero-Shot | ❌ | ❌ | ⚡⚡⚡ |
| Basic RAG | ❌ | ❌ | ⚡⚡⚡ |
| Chain-of-Thought | ✅ | ❌ | ⚡⚡ |
| Self-Consistency | ✅ | ✅ | ⚡ |
| ReAct | ✅ | ❌ | ⚡ |

## Best Practices

1. **Use CoT for complex questions**: Multi-step reasoning benefits from explicit steps
2. **Self-consistency for critical tasks**: Higher accuracy at cost of speed
3. **Clear instructions**: Be explicit about expected output format
4. **Temperature tuning**: Lower for factual, higher for creative

## Next Steps

- [⚙️ Building Pipelines](pipelines.md) - End-to-end systems
- [📊 Evaluation](evaluation.md) - Measure performance
