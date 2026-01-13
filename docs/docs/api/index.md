# API Reference

Below is an overview of the modules, classes, and functions available in **Rankify v0.1.4**.

## Dataset Module
- [Dataset](dataset.md)

## Metrics Module
- [Metrics](metrics.md)

## Retrievers
- [Retriever](retrievers/retriever.md) - Unified retriever interface
- [BM25 Retriever](retrievers/bm25.md) - Sparse retrieval
- [Dense Retriever (DPR)](retrievers/dense.md) - Dense Passage Retrieval
- [ANCE Retriever](retrievers/ance.md) - Approximate Nearest Neighbor Negative Contrastive Estimation
- [BGE Retriever](retrievers/bge.md) - BAAI General Embedding
- [ColBERT Retriever](retrievers/colbert.md) - Late interaction retrieval
- [Contriever Retriever](retrievers/contriever.md) - Contrastive retriever
- [Online Retriever](retrievers/online.md) - Real-time web retrieval
- [HyDE Retriever](retrievers/hyde.md) - Hypothetical Document Embeddings

## Indexing Module
- [Indexing](indexing.md) - Build custom search indices

## Rerankers

### Main Interface
- [Reranking](rerankings/reranking.md) - Unified reranking interface
- [Base](rerankings/base.md) - Base reranker class

### Pointwise Rerankers
- [MonoBERT](rerankings/monobert.md) - BERT-based pointwise reranker
- [MonoT5](rerankings/monot5.md) - T5-based pointwise reranker
- [UPR](rerankings/upr.md) - Unsupervised Passage Reranker

### Pairwise Rerankers
- [RankGPT](rerankings/rankgpt.md) - GPT-based pairwise reranking
- [InRanker](rerankings/inrank.md) - Instruction-based reranker
- [EchoRank](rerankings/echo_rank.md) - Echo-based reranking

### Listwise Rerankers
- [RankT5](rerankings/rankt5.md) - T5-based listwise reranker
- [ListT5](rerankings/listt5.md) - Listwise T5 reranker
- [LiT5](rerankings/lit5.md) - Lightweight T5 reranker
- [Transformer Reranker](rerankings/transformer_reranker.md) - Cross-encoder rerankers

### API-Based Rerankers
- [API Reranker](rerankings/apiranker.md) - Cohere, Jina, Voyage, MixedBread.ai

### LLM-Based Rerankers
- [First Reranker](rerankings/first_ranker.md) - First model reranker
- [Incontext Reranker](rerankings/incontext_reranker.md) - In-context learning reranker
- [Vicuna Reranker](rerankings/vicuna_reranker.md) - Vicuna-based reranker
- [Zephyr Reranker](rerankings/zephyr_reranker.md) - Zephyr-based reranker

### Embedding-Based Rerankers
- [ColBERT Reranker](rerankings/colbert_ranker.md) - ColBERT for reranking
- [Sentence Transformer](rerankings/sentence_transformer_reranker.md) - Sentence transformer reranker
- [SPLADE Reranker](rerankings/splade_reranker.md) - Sparse reranker
- [LLM2Vec Reranker](rerankings/llm2vec_reranker.md) - LLM to vector reranker
- [LLM Layerwise](rerankings/llm_layerwise_ranker.md) - Layerwise LLM reranker

### Other Rerankers
- [FlashRank](rerankings/flashrank.md) - Fast ONNX-based reranker
- [Blender Reranker](rerankings/blender.md) - PairRM blender
- [TwoLAR](rerankings/twolar.md) - Two-stage listwise reranker
- [Rank FiD](rerankings/rank_fid.md) - Fusion-in-Decoder reranker

## Generator Module

### Main Interface
- [Generator](generators/generator.md) - Unified generator interface
- [Prompt Generator](generators/prompt_generator.md)
- [Prompt Template](generators/prompt_template.md)

### LLM Endpoints
- [Base RAG Model](generators/models/base_rag_model.md)
- [Model Factory](generators/models/model_factory.md)
- [HuggingFace Model](generators/models/huggingface_model.md)
- [LiteLLM Model](generators/models/litellm_model.md)
- [OpenAI Model](generators/models/openai_model.md)
- [vLLM Model](generators/models/vllm_model.md)
- [FiD Model](generators/models/fid_model.md)

### RAG Methods
- [Base RAG Method](generators/rag_methods/basic_rag_method.md)
- [Zero-Shot RAG](generators/rag_methods/zero_shot.md)
- [Basic RAG](generators/rag_methods/basic_rag.md)
- [Chain-of-Thought RAG](generators/rag_methods/chain_of_thought_rag.md)
- [Self-Consistency RAG](generators/rag_methods/self_consistency_rag.md)
- [ReAct RAG](generators/rag_methods/react_rag.md)
- [Fusion-in-Decoder](generators/rag_methods/fid_rag_method.md)
- [In-Context RALM](generators/rag_methods/in_context_ralm_rag.md)

## Tools Module
- [Tools](tools.md) - WebSearchTool and agent utilities