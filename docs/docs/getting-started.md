---
title: "Getting Started"
sidebar_position: 2
---

# 🚀 Getting Started with Rankify

## 🔥 Overview
Rankify is a **powerful Python toolkit** for **Retrieval, Re-Ranking, and Retrieval-Augmented Generation (RAG)**. It integrates **7 retrieval techniques, 24 state-of-the-art re-ranking models, and multiple RAG methods**, enabling seamless experimentation across retrieval pipelines.



---

## 📚 Using Pre-Retrieved Datasets
Rankify includes **pre-retrieved datasets** from **Hugging Face**:
🔗 [Hugging Face Dataset Repository](https://huggingface.co/datasets/abdoelsayed/reranking-datasets-light)

### **1️⃣ Load a Pre-Retrieved Dataset**
```python
from rankify.dataset.dataset import Dataset
dataset = Dataset(retriever="bm25", dataset_name="nq-dev", n_docs=100)
documents = dataset.download(force_download=False)
```
### **2️⃣ Available Retrieval Methods**
- `bm25`, `dpr`, `ance`, `colbert`, `bge`, `contriever`
- **Example**:

```python
dataset = Dataset(retriever="bge", dataset_name="nq-dev", n_docs=100)
documents = dataset.download(force_download=False)
```

---

## 🔍 Performing Retrieval
Rankify supports **BM25, DPR, ANCE, ColBERT, BGE, and Contriever** for retrieval.

### **1️⃣ Example: Retrieving Documents**

```python
from rankify.dataset.dataset import Document, Question
from rankify.retrievers.retriever import Retriever

documents = [Document(question=Question("Who wrote Hamlet?"))]

retriever = Retriever(method="bm25", n_docs=5, index_type="wiki")
retrieved_docs = retriever.retrieve(documents)

for doc in retrieved_docs:
    print(doc)
```

### **2️⃣ Using Different Retrieval Models**

```python
retriever = Retriever(method="colbert", model="colbert-ir/colbertv2.0", n_docs=5, index_type="wiki")
retriever = Retriever(method="dpr", model="dpr-multi", n_docs=5, index_type="msmarco")
```

---

## 📊 Running Re-Ranking
Rankify provides **multiple re-ranking models**.

### **1️⃣ Example: Using MonoT5 for Re-Ranking**

```python
from rankify.dataset.dataset import Document, Question, Context
from rankify.models.reranking import Reranking

question = Question("When did Thomas Edison invent the light bulb?")
contexts = [Context(text="Thomas Edison invented the light bulb in 1879.")]

document = Document(question=question, contexts=contexts)

reranker = Reranking(method="monot5", model_name="monot5-base-msmarco")
reranker.rank([document])

for context in document.reorder_contexts:
    print(context.text)
```

---

### **2️⃣ Other Available Re-Rankers**
- **Cross-Encoders**
- **MonoT5, MonoBERT, RankT5, ListT5**
- **ColBERT Ranker**
- **LLM-based Re-rankers (Vicuna, Zephyr, GPT)**

---

## 🤖 Retrieval-Augmented Generation (RAG)
### **1️⃣ Example: Using GPT for RAG**

```python
from rankify.dataset.dataset import Document, Question, Context
from rankify.generator.generator import Generator

question = Question("What is the capital of France?")
contexts = [Context(text="The capital of France is Paris.")]

document = Document(question=question, contexts=contexts)

generator = Generator(method="in-context-ralm", model_name='meta-llama/Llama-3.1-8B')
generated_answers = generator.generate([document])

print(generated_answers)
```

---

## 📊 Evaluating Models
Rankify provides **retrieval, re-ranking, and generation evaluation metrics**.

### **1️⃣ Evaluate Retrieval Performance**
from rankify.metrics.metrics import Metrics

```python
metrics = Metrics(documents)
retrieval_results = metrics.calculate_retrieval_metrics(ks=[1, 5, 10], use_reordered=False)
print(retrieval_results)

### **2️⃣ Evaluate Re-Ranked Results**
re_ranked_results = metrics.calculate_retrieval_metrics(ks=[1, 5, 10], use_reordered=True)
print(re_ranked_results)

### **3️⃣ Evaluate RAG Performance**
generated_answers = generator.generate(documents)
generation_metrics = metrics.calculate_generation_metrics(generated_answers)
print(generation_metrics)
```

---
