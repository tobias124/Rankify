---
title: "Rankify Server - REST API"
---

# 🌐 Rankify Server

Deploy Rankify as a REST API for production applications.

## Quick Start

### Command Line

```bash
# Start server with default config
rankify serve --port 8000

# Custom configuration
rankify serve --retriever bge --reranker flashrank --port 8000
```

### Python

```python
from rankify.server import RankifyServer

server = RankifyServer(
    retriever="bge",
    reranker="flashrank",
    generator="basic-rag",
)

server.start(host="0.0.0.0", port=8000)
```

---

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "retriever": "bge",
  "reranker": "flashrank",
  "generator": "basic-rag"
}
```

---

### Retrieve Documents

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "n_docs": 10}'
```

**Response:**
```json
{
  "query": "What is machine learning?",
  "documents": [
    {
      "id": "doc_1",
      "text": "Machine learning is a subset of AI...",
      "title": "Introduction to ML",
      "score": 0.89
    }
  ],
  "latency_ms": 45.2
}
```

---

### Rerank Documents

```bash
curl -X POST http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is deep learning?",
    "documents": [
      {"id": "1", "text": "Deep learning uses neural networks..."},
      {"id": "2", "text": "Machine learning is a type of AI..."},
      {"id": "3", "text": "Deep neural networks have many layers..."}
    ],
    "top_k": 2
  }'
```

**Response:**
```json
{
  "query": "What is deep learning?",
  "documents": [
    {"id": "3", "text": "Deep neural networks have many layers...", "score": 0.92},
    {"id": "1", "text": "Deep learning uses neural networks...", "score": 0.87}
  ],
  "latency_ms": 12.5
}
```

---

### RAG Generation

```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain transformers", "n_contexts": 5}'
```

**Response:**
```json
{
  "query": "Explain transformers",
  "answer": "Transformers are a neural network architecture that uses self-attention...",
  "contexts": [
    {"id": "1", "text": "The transformer was introduced in...", "score": 0.95}
  ],
  "latency_ms": 1250.3
}
```

---

### Batch Retrieve

```bash
curl -X POST http://localhost:8000/retrieve/batch \
  -H "Content-Type: application/json" \
  -d '["What is AI?", "What is ML?", "What is DL?"]'
```

---

## Server Configuration

```python
from rankify.server import RankifyServer

server = RankifyServer(
    retriever="bge",              # Retriever method
    reranker="flashrank",         # Reranker method
    generator="basic-rag",        # RAG method (optional)
    retriever_model=None,         # Specific retriever model
    reranker_model="ms-marco-MiniLM-L-12-v2",
    generator_model="gpt-4o-mini",
    generator_backend="openai",
    index_type="wiki",
    n_docs=100,
)

server.start(
    host="0.0.0.0",
    port=8000,
    workers=4,      # Number of workers
    reload=False,   # Auto-reload for dev
)
```

---

## Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
RUN pip install rankify fastapi uvicorn

EXPOSE 8000

CMD ["python", "-m", "rankify.server", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t rankify-server .
docker run -p 8000:8000 rankify-server
```

---

## OpenAPI Documentation

Access interactive API docs at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Client Libraries

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/rag",
    json={"query": "What is AI?", "n_contexts": 5}
)
print(response.json()["answer"])
```

### JavaScript

```javascript
const response = await fetch('http://localhost:8000/rag', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({query: 'What is AI?', n_contexts: 5})
});
const data = await response.json();
console.log(data.answer);
```
