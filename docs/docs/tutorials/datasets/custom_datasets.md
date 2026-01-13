---
title: "Creating Custom Datasets"
---

# 🛠 Creating Custom Datasets

Build datasets from your own QA data.

## Document Structure

```python
from rankify.dataset.dataset import Document, Question, Answer, Context

# Create a document
doc = Document(
    question=Question("What is Python?"),
    answers=Answer(["A programming language", "A high-level language"]),
    contexts=[
        Context(
            text="Python is a high-level programming language.",
            id="1",
            title="Python (programming)",
            score=0.95
        ),
    ]
)
```

## From JSON/JSONL

```python
import json

# Load custom data
with open("my_data.jsonl") as f:
    data = [json.loads(line) for line in f]

documents = []
for item in data:
    doc = Document(
        question=Question(item["question"]),
        answers=Answer(item["answers"]),
        contexts=[
            Context(text=ctx["text"], id=ctx["id"])
            for ctx in item.get("contexts", [])
        ]
    )
    documents.append(doc)
```

## Saving Datasets

```python
from rankify.dataset.dataset import Dataset

# Save to file
Dataset.save_dataset(documents, "./my_dataset.json")

# Load from file
loaded = Dataset.load_dataset("./my_dataset.json", n_docs=100)
```
