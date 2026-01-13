---
title: "Large-Scale Applications"
---

# 🔌 Using Rankify in Large-Scale Applications

Best practices for production deployments.

## Batch Processing

```python
from tqdm import tqdm
import torch

def batch_rerank(documents, batch_size=32):
    """Process documents in batches."""
    results = []
    
    reranker = Reranking(method="monot5", model_name="monot5-base-msmarco")
    
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:i+batch_size]
        reranked = reranker.rank(batch)
        results.extend(reranked)
        
        # Clear GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results
```

## Multi-GPU Processing

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Models will automatically use available GPUs
```

## Memory Management

```python
# Use context managers for memory-intensive operations
with torch.inference_mode():
    results = reranker.rank(large_batch)
```
