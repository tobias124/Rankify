---
title: "Loading & Saving Datasets"
---

# 📥 Loading & Saving Datasets

Work with datasets in various formats.

## Loading from Hugging Face

```python
from rankify.dataset.dataset import Dataset

dataset = Dataset(retriever="bm25", dataset_name="nq-dev", n_docs=100)
documents = dataset.download(force_download=False)
```

## Loading from File

```python
documents = Dataset.load_dataset("./path/to/data.json", n_docs=100)
```

## Saving Documents

```python
Dataset.save_dataset(documents, "./output.json")
```

## Format Conversion

```python
# Convert to pandas
import pandas as pd

data = []
for doc in documents:
    data.append({
        "question": doc.question.question,
        "answers": doc.answers.answers,
        "num_contexts": len(doc.contexts)
    })

df = pd.DataFrame(data)
df.to_csv("dataset_summary.csv")
```
