---
title: "Cloud & GPUs"
---

# 🖥️ Running Rankify on Cloud & GPUs

Deploy Rankify on cloud infrastructure.

## GPU Setup

```python
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

# Set specific device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
```

## vLLM for Fast Inference

```python
# Install with reranking extras
# pip install "rankify[reranking]"

from rankify.generator.generator import Generator

generator = Generator(
    method="basic-rag",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    backend="vllm"
)
```

## Cloud Deployment

### Docker

```dockerfile
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN pip install "rankify[all]"

COPY app.py /app/
CMD ["python", "/app/app.py"]
```

### Environment Variables

```bash
export RERANKING_CACHE_DIR=/data/cache
export CUDA_VISIBLE_DEVICES=0,1
export OPENAI_API_KEY=your-key
```
