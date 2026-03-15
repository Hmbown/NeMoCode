# Generative AI Examples

> Reference architectures and sample implementations for RAG, agents, and AI workflows.

## Canonical URL

| Resource | URL |
|---|---|
| GitHub | `https://github.com/NVIDIA/GenerativeAIExamples` |

## Overview

NVIDIA's Generative AI Examples repository provides production-ready reference implementations for common AI application patterns. These examples demonstrate how to combine NIM, NeMo, and partner technologies into working applications.

## Example Categories

### RAG (Retrieval-Augmented Generation)
- Basic RAG pipeline with NIM + vector database
- Multi-turn conversational RAG
- Multi-document RAG with source attribution
- Structured data RAG (SQL, tables)

### Agentic RAG
- RAG with dynamic tool selection
- Multi-step reasoning over retrieved context
- Self-correcting RAG (query refinement on failure)

### Human-in-the-Loop Agents
- Agents that request human approval for critical actions
- Interactive agent workflows with user feedback
- Supervised autonomous task execution

### Tool Calling with LangGraph
- LangGraph-based agent with NIM backend
- Multi-tool agent orchestration
- Stateful conversation with tool persistence

### Knowledge Graph RAG
- RAG over knowledge graphs (Neo4j, etc.)
- Graph-enhanced retrieval for complex queries
- Entity-relationship aware generation

### Vision Workflows
- Document understanding with VLMs
- Image-based RAG (visual + text retrieval)
- Multimodal agent workflows

### Safer AI (NeMo Auditor)
- Automated red-teaming pipelines
- Safety benchmark evaluation
- Bias detection workflows

## Technology Stack Used

| Component | Technology |
|---|---|
| LLM Inference | NIM for LLMs |
| Embedding | NeMo Retriever NIM |
| Reranking | NeMo Retriever NIM |
| Vector DB | Milvus, pgvector, or FAISS |
| Orchestration | LangChain, LlamaIndex, or LangGraph |
| Safety | NeMo Guardrails |
| Frontend | Gradio or Streamlit |

## Running Examples

Most examples follow this pattern:

```bash
git clone https://github.com/NVIDIA/GenerativeAIExamples
cd GenerativeAIExamples/<example-name>
export NVIDIA_API_KEY="nvapi-xxx"
docker compose up
```
