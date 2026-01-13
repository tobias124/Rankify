# Indexing Module

The indexing module provides tools for creating custom search indices for various retrieval methods.

## Overview

Rankify supports building indices for:
- **BM25** (Lucene-based sparse retrieval)
- **DPR** (Dense Passage Retrieval)
- **ANCE** (Approximate Nearest Neighbor Negative Contrastive Estimation)
- **BGE** (BAAI General Embedding)
- **ColBERT** (Contextualized Late Interaction over BERT)
- **Contriever** (Contrastive Retriever)

## CLI Usage

Use the `rankify-index` command to build indices:

```bash
# BM25 Index
rankify-index index data/corpus.jsonl --retriever bm25 --output ./indices

# Dense Retrievers
rankify-index index data/corpus.jsonl --retriever dpr --device cuda --batch_size 16

# See all options
rankify-index index --help
```

## API Reference

::: rankify.indexing
    options:
        show_source: true
        members: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
