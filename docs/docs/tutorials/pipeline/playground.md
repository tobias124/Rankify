---
title: "Web Playground - Interactive UI"
---

# 🎨 Web Playground

Interactive Gradio interface for experimenting with Rankify models.

## Quick Start

```python
from rankify.ui import launch_playground

# Launch interactive playground
launch_playground(port=7860)
```

Then open `http://localhost:7860` in your browser.

---

## Features

### 🔍 Search Tab
Test different retrievers interactively:
- Enter your query
- Select retriever (BM25, BGE, ColBERT, etc.)
- View ranked results
- Copy generated code

### 📊 Rerank Tab
Test rerankers on your documents:
- Enter query and documents
- Select reranker method
- See reranked results with scores
- Export code

### 🤖 RAG Tab
Full RAG pipeline testing:
- Enter question
- Configure retriever + reranker + RAG method
- Generate answers with context
- View source documents

### 💻 Code Generator
Generate code for any configuration:
- Select task type
- Choose models
- Get ready-to-use Python code

---

## Screenshot

```
┌──────────────────────────────────────────────────────────────┐
│  🚀 Rankify Playground                                       │
├──────────────────────────────────────────────────────────────┤
│  [🔍 Search] [📊 Rerank] [🤖 RAG] [💻 Code Generator]        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Query: [What is machine learning?_____________]             │
│                                                              │
│  Retriever: [bge ▼]     N Docs: [50 ──●──── 100]            │
│                                                              │
│  [🔍 Search]                                                 │
│                                                              │
│  ─────────────────────────────────────────────────────────   │
│  Results:                                                    │
│  1. Introduction to Machine Learning                         │
│     Machine learning is a subset of AI that enables...       │
│                                                              │
│  2. Types of ML Algorithms                                   │
│     There are three main types: supervised, unsupervised...  │
│                                                              │
│  ─────────────────────────────────────────────────────────   │
│  Code:                                                       │
│  ```python                                                   │
│  from rankify import pipeline                                │
│  search = pipeline("search", retriever="bge", n_docs=50)     │
│  results = search("What is machine learning?")               │
│  ```                                                         │
└──────────────────────────────────────────────────────────────┘
```

---

## Configuration

```python
from rankify.ui import launch_playground

launch_playground(
    port=7860,           # Port to run on
    share=True,          # Create public Gradio link
    server_name="0.0.0.0",  # Bind address
)
```

---

## Use Cases

1. **Quick Testing** - Try models before writing code
2. **Demo to Stakeholders** - Show capabilities visually
3. **Model Comparison** - Test different configurations
4. **Code Generation** - Generate starter code
5. **Debugging** - Understand model behavior

---

## Requirements

Install Gradio:
```bash
pip install gradio
```

---

## Customization

Create a custom playground:

```python
from rankify.ui.playground import create_playground_app

# Get the Gradio app
app = create_playground_app()

# Add custom components
with app:
    gr.Markdown("## My Custom Section")
    # ... add more

# Launch
app.launch()
```
