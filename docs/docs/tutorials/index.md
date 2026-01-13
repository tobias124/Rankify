---
title: "Tutorials"
sidebar_position: 5
---

# 📚 Tutorials & Guides  

Welcome to Rankify tutorials! These guides cover everything from basic usage to advanced customization.

## 🚀 Retrieval & Search Techniques

| Tutorial | Description |
|----------|-------------|
| [📌 Introduction to Information Retrieval](retrieval/introduction.md) | Overview of retrieval in Rankify |
| [🔍 Using Sparse Retrievers (BM25)](retrieval/bm25.md) | BM25 sparse retrieval |
| [🧠 Using Dense Retrievers](retrieval/dense_retrievers.md) | DPR, ANCE, ColBERT, BGE, Contriever |
| [🤖 Hybrid Retrieval](retrieval/hybrid.md) | Combining sparse & dense methods |
| [📂 Prebuilt Corpora & Indexes](retrieval/prebuilt_corpora.md) | Using Wikipedia and MS MARCO indices |
| [🔎 Custom Datasets & Indexing](retrieval/custom_datasets.md) | Building your own indices |

## 📊 Re-Ranking Strategies

| Tutorial | Description |
|----------|-------------|
| [📌 Introduction to Re-Ranking](reranking/introduction.md) | Overview of 23 reranking methods |
| [🎯 Pointwise Re-Ranking](reranking/pointwise.md) | MonoBERT, MonoT5, UPR, FlashRank |
| [🔄 Pairwise Re-Ranking](reranking/pairwise.md) | RankGPT, InRanker, EchoRank |
| [📃 Listwise Re-Ranking](reranking/listwise.md) | RankT5, LiT5, Transformer Rankers |
| [🦾 API-Based Rerankers](reranking/api_rerankers.md) | Voyage, Jina, MixedBread.ai |
| [📈 Comparing Performance](reranking/evaluation.md) | Benchmarking rerankers |

## 🧠 Retrieval-Augmented Generation (RAG)

| Tutorial | Description |
|----------|-------------|
| [📌 Introduction to RAG](rag/introduction.md) | Overview of 7 RAG methods |
| [📥 Zero-Shot RAG](rag/zero_shot.md) | GPT, LLaMA, vLLM backends |
| [🔁 Fusion-in-Decoder (FiD)](rag/fid.md) | FiD architecture |
| [📄 In-Context Learning](rag/in_context.md) | Chain-of-Thought, Self-Consistency, ReAct |
| [⚙️ Building RAG Pipelines](rag/pipelines.md) | End-to-end systems |
| [📊 Evaluating RAG Models](rag/evaluation.md) | EM, F1, BLEU metrics |

## 📂 Working with Datasets

| Tutorial | Description |
|----------|-------------|
| [📌 Prebuilt Benchmark Datasets](datasets/benchmark_datasets.md) | NQ, TriviaQA, SQuAD, etc. |
| [🛠 Creating Custom Datasets](datasets/custom_datasets.md) | Build from your data |
| [📥 Loading & Saving](datasets/loading_saving.md) | Dataset I/O |
| [📊 Dataset Evaluation](datasets/evaluation.md) | Evaluate retrieval quality |

## 🛠 Evaluation & Benchmarking

| Tutorial | Description |
|----------|-------------|
| [📏 Retrieval Metrics](evaluation/retrieval_metrics.md) | Recall@k, MRR, P@k |
| [📈 Reranking Metrics](evaluation/reranking_metrics.md) | NDCG, MAP |
| [🧠 RAG Metrics](evaluation/rag_metrics.md) | Exact Match, F1, Contains |
| [📊 Method Comparisons](evaluation/comparisons.md) | Systematic benchmarking |

## ⚡ Advanced Usage & Customization

| Tutorial | Description |
|----------|-------------|
| [🛠 Custom Retrievers](advanced/custom_retrievers.md) | Extend BaseRetriever |
| [🔧 Custom Rerankers](advanced/custom_rerankers.md) | Extend BaseRanking |
| [⚙️ Custom RAG Models](advanced/custom_rag.md) | Create new RAG methods |
| [💾 Saving & Loading](advanced/saving_models.md) | Model persistence |

## 🚀 Deployment & Integration

| Tutorial | Description |
|----------|-------------|
| [🔌 Large-Scale Applications](deployment/large_scale.md) | Batch processing, multi-GPU |
| [🌍 External APIs](deployment/apis.md) | OpenAI, Cohere, LiteLLM |
| [🖥️ Cloud & GPUs](deployment/cloud.md) | vLLM, Docker, cloud deployment |
| [🐞 Debugging](deployment/debugging.md) | Logging, profiling |

---

## Quick Links

- [📖 Getting Started](../getting-started.md) - First steps with Rankify
- [📚 API Reference](../api/index.md) - Complete API documentation
- [🔧 Installation](../installation.md) - Setup guide
