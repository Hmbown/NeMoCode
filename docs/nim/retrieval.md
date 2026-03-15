# NIM for Retrieval (Embedding & Reranking)

> Text embedding and reranking microservices for RAG pipelines.

## Canonical URLs

| Resource | URL |
|---|---|
| Text Embedding NIM | `https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/` |
| Text Reranking NIM | `https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/` |

## Overview

NeMo Retriever NIMs provide the retrieval components for Retrieval-Augmented Generation (RAG) pipelines. They include text embedding models for vector search and reranking models for result refinement.

## Text Embedding NIM

### API Compatibility
OpenAI-compatible `/v1/embeddings` endpoint:

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/nv-embedqa-e5-v5",
    "input": ["What is NeMo?"]
  }'
```

### Features
- OpenAI-compatible REST API
- gRPC API for high-throughput
- Batch embedding support
- Multiple embedding dimensions
- Passage and query embedding modes

### Available Models
- NVIDIA NV-EmbedQA models (various sizes)
- Optimized for retrieval quality and speed

## Text Reranking NIM

### API
```bash
curl -X POST http://localhost:8000/v1/ranking \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/nv-rerankqa-mistral-4b-v3",
    "query": {"text": "What is NeMo?"},
    "passages": [
      {"text": "NeMo is NVIDIA AI framework..."},
      {"text": "NeMo is a fish species..."}
    ]
  }'
```

### Features
- Score and reorder candidate passages
- Dramatically improves RAG quality
- Low latency for real-time pipelines

## RAG Pipeline Pattern

```
Query → Embedding NIM → Vector DB Search → Reranking NIM → Top-K Results → LLM NIM → Answer
```

1. **Embed** the query using Embedding NIM
2. **Search** a vector database (Milvus, Pinecone, pgvector, etc.)
3. **Rerank** retrieved candidates using Reranking NIM
4. **Generate** answer using LLM NIM with reranked context
