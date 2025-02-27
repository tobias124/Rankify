---
title: "Installation Guide"
---

## 🔧 Installation  

#### Set up the virtual environment
First, create and activate a conda environment with Python 3.10:

```bash
conda create -n rankify python=3.10
conda activate rankify
```
#### Install PyTorch 2.5.1
We recommend installing Rankify with PyTorch 2.5.1 for Rankify. Refer to the [PyTorch installation page](https://pytorch.org/get-started/previous-versions/) for platform-specific installation commands. 

If you have access to GPUs, it's recommended to install the CUDA version 12.4 or 12.6 of PyTorch, as many of the evaluation metrics are optimized for GPU use.

To install Pytorch 2.5.1 you can install it from the following command:
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

#### Basic Installation
To install **Rankify**, simply use **pip** (requires Python 3.10+):  
```bash
pip install rankify
```
This will install the base functionality required for retrieval, re-ranking, and retrieval-augmented generation (RAG).  

#### Recommended Installation  
For full functionality, we **recommend installing Rankify with all dependencies**:
```bash
pip install "rankify[all]"
```
This ensures you have all necessary modules, including retrieval, re-ranking, and RAG support.

#### Optional Dependencies
If you prefer to install only specific components, choose from the following:
```bash
# Install dependencies for retrieval only (BM25, DPR, ANCE, etc.)
pip install "rankify[retriever]"

# Install base re-ranking with vLLM support for `FirstModelReranker`, `LiT5ScoreReranker`, `LiT5DistillReranker`, `VicunaReranker`, and `ZephyrReranker'.
pip install "rankify[reranking]"
```

Or, to install from **GitHub** for the latest development version:  
```bash
git clone https://github.com/DataScienceUIBK/rankify.git
cd rankify
pip install -e .
# For full functionality we recommend installing Rankify with all dependencies:
pip install -e ".[all]"
# Install dependencies for retrieval only (BM25, DPR, ANCE, etc.)
pip install -e ".[retriever]"
# Install base re-ranking with vLLM support for `FirstModelReranker`, `LiT5ScoreReranker`, `LiT5DistillReranker`, `VicunaReranker`, and `ZephyrReranker'.
pip install -e ".[reranking]"
```

#### Using ColBERT Retriever  
If you want to use **ColBERT Retriever**, follow these additional setup steps:
```bash
# Install GCC and required libraries
conda install -c conda-forge gcc=9.4.0 gxx=9.4.0
conda install -c conda-forge libstdcxx-ng
```
```bash
# Export necessary environment variables
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CC=gcc
export CXX=g++
export PATH=$CONDA_PREFIX/bin:$PATH

# Clear cached torch extensions
rm -rf ~/.cache/torch_extensions/*
```