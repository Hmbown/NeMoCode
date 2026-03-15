# Embedding Models

> Models for converting text and multimodal content into vector representations.

## Canonical URLs

| Resource | URL |
|---|---|
| Text Embedding NIM | `https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/` |
| Cosmos Embed1 | `https://docs.nvidia.com/nim/cosmos/latest/` |
| NV-CLIP | `https://docs.nvidia.com/nim/nvclip/latest/` |
| build.nvidia.com | `https://build.nvidia.com/explore/discover` (filter: Retrieval) |

## Text Embedding Models

### NV-EmbedQA Series

| Model | Description |
|---|---|
| **NV-EmbedQA-E5-v5** | Text embedding optimized for QA retrieval |
| **NV-EmbedQA-Mistral-7B-v0.1** | Large embedding model based on Mistral 7B |

### Features
- High-quality dense vector representations
- Optimized for retrieval tasks (QA, search, RAG)
- Multiple embedding dimensions available
- Passage and query embedding modes (asymmetric)
- Batch processing support

### API (OpenAI-compatible)

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/nv-embedqa-e5-v5",
    "input": ["What is NeMo?", "NeMo is NVIDIA framework."],
    "encoding_format": "float"
  }'
```

### Response Format

```json
{
  "data": [
    {"embedding": [0.012, -0.034, ...], "index": 0},
    {"embedding": [0.045, -0.012, ...], "index": 1}
  ],
  "model": "nvidia/nv-embedqa-e5-v5",
  "usage": {"prompt_tokens": 12, "total_tokens": 12}
}
```

## Multimodal Embedding Models

### Cosmos Embed1
- Video and image embeddings for world foundation models
- Designed for physical AI and robotics applications
- Embeds visual scenes into semantic vector space

### NV-CLIP
- Vision-language embeddings based on CLIP architecture
- Zero-shot image classification
- Image-text similarity scoring
- Cross-modal retrieval (text→image, image→text)

### NV-DINOv2
- Self-supervised vision embeddings
- Few-shot visual classification
- Feature extraction for downstream vision tasks

## Use Cases

| Use Case | Recommended Model |
|---|---|
| Text search / RAG | NV-EmbedQA-E5-v5 |
| QA retrieval | NV-EmbedQA-Mistral-7B-v0.1 |
| Image classification | NV-CLIP |
| Image search | NV-CLIP |
| Video understanding | Cosmos Embed1 |
| Visual features | NV-DINOv2 |

## Vector Database Compatibility

Embeddings work with any vector database:
- Milvus
- Pinecone
- Weaviate
- pgvector (PostgreSQL)
- FAISS
- Qdrant
- Chroma
