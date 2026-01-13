---
title: "Custom Datasets & Indexing"
---

# 🔎 Custom Datasets & Indexing

Learn how to create custom search indices for your own document collections.

## Corpus Format

Prepare your corpus as a JSONL file with the following format:

```json
{"id": "doc1", "title": "Albert Einstein", "text": "Albert Einstein was a German-born theoretical physicist..."}
{"id": "doc2", "title": "Isaac Newton", "text": "Sir Isaac Newton was an English mathematician..."}
{"id": "doc3", "title": "Marie Curie", "text": "Marie Curie was a Polish-born physicist and chemist..."}
```

**Required fields:**
- `id`: Unique document identifier
- `text`: Document content

**Optional fields:**
- `title`: Document title (recommended)

## CLI Indexing

### BM25 Index

```bash
rankify-index index data/my_corpus.jsonl \
    --retriever bm25 \
    --output ./my_indices \
    --threads 32
```

### DPR Index

```bash
rankify-index index data/my_corpus.jsonl \
    --retriever dpr \
    --encoder facebook/dpr-ctx_encoder-single-nq-base \
    --batch_size 16 \
    --device cuda \
    --output ./my_indices
```

### ANCE Index

```bash
rankify-index index data/my_corpus.jsonl \
    --retriever ance \
    --encoder castorini/ance-dpr-context-multi \
    --batch_size 16 \
    --device cuda \
    --output ./my_indices
```

### BGE Index

```bash
rankify-index index data/my_corpus.jsonl \
    --retriever bge \
    --encoder BAAI/bge-large-en-v1.5 \
    --batch_size 16 \
    --device cuda \
    --output ./my_indices
```

### Contriever Index

```bash
rankify-index index data/my_corpus.jsonl \
    --retriever contriever \
    --encoder facebook/contriever-msmarco \
    --batch_size 16 \
    --device cuda \
    --output ./my_indices
```

### ColBERT Index

```bash
rankify-index index data/my_corpus.jsonl \
    --retriever colbert \
    --batch_size 32 \
    --device cuda \
    --output ./my_indices
```

## Using Custom Indices

After building, use your index with the `index_folder` parameter:

```python
from rankify.retrievers.retriever import Retriever
from rankify.dataset.dataset import Document, Question

# Point to your custom index
retriever = Retriever(
    method="bm25",
    n_docs=10,
    index_folder="./my_indices/my_corpus/bm25_index"
)

# Query
documents = [Document(question=Question("What did Einstein discover?"))]
results = retriever.retrieve(documents)

for ctx in results[0].contexts:
    print(f"[{ctx.score:.4f}] {ctx.title}: {ctx.text[:100]}...")
```

## Python API for Indexing

You can also build indices programmatically:

```python
from rankify.indexing.lucene_indexer import LuceneIndexer
from rankify.indexing.dpr_indexer import DPRIndexer

# BM25 Index
bm25_indexer = LuceneIndexer(
    corpus_path="data/my_corpus.jsonl",
    output_dir="./my_indices",
    threads=32
)
bm25_indexer.build_index()

# DPR Index
dpr_indexer = DPRIndexer(
    corpus_path="data/my_corpus.jsonl",
    output_dir="./my_indices",
    encoder_name="facebook/dpr-ctx_encoder-single-nq-base",
    batch_size=16,
    device="cuda"
)
dpr_indexer.build_index()
```

## Index Output Structure

```
my_indices/
└── my_corpus/
    ├── bm25_index/
    │   └── lucene/
    ├── dpr_index_wiki/
    │   ├── embeddings.npy
    │   └── doc_ids.json
    ├── ance_index_wiki/
    ├── bge_index_wiki/
    ├── colbert_index_wiki/
    └── contriever_index_wiki/
```

## Best Practices

1. **Chunk large documents**: Split documents into 100-200 word passages
2. **Clean text**: Remove HTML, special characters, and normalize whitespace
3. **Use GPU for dense indexing**: Significantly speeds up encoding
4. **Test on sample first**: Build index on small sample before full corpus

## Memory Requirements

| Method | RAM per 1M docs | GPU RAM |
|--------|-----------------|---------|
| BM25 | ~2GB | N/A |
| DPR | ~8GB | 8GB |
| ANCE | ~8GB | 8GB |
| BGE | ~8GB | 8GB |
| ColBERT | ~20GB | 16GB |

## Next Steps

- [📌 Introduction](introduction.md) - Overview of retrieval methods
- [📊 Evaluation](../evaluation/retrieval_metrics.md) - Evaluate your index
