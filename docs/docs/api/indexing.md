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

### Base Indexer
::: rankify.indexing.base_indexer
    handler: python
    options:
        show_source: true
        heading_level: 3

### Lucene Indexer (BM25)
::: rankify.indexing.lucene_indexer
    handler: python
    options:
        show_source: true
        heading_level: 3

### DPR Indexer
::: rankify.indexing.dpr_indexer
    handler: python
    options:
        show_source: true
        heading_level: 3

### ANCE Indexer
::: rankify.indexing.ance_indexer
    handler: python
    options:
        show_source: true
        heading_level: 3

### BGE Indexer
::: rankify.indexing.bge_indexer
    handler: python
    options:
        show_source: true
        heading_level: 3

### ColBERT Indexer
::: rankify.indexing.colbert_indexer
    handler: python
    options:
        show_source: true
        heading_level: 3

### Contriever Indexer
::: rankify.indexing.contriever_indexer
    handler: python
    options:
        show_source: true
        heading_level: 3
