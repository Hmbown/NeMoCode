# Vision Models

> Models for image understanding, classification, and visual reasoning.

## Canonical URLs

| Resource | URL |
|---|---|
| NV-CLIP NIM | `https://docs.nvidia.com/nim/nvclip/latest/` |
| NeMo VFM | NeMo Framework User Guide (VFM section) |
| build.nvidia.com | `https://build.nvidia.com/explore/discover` (filter: Vision) |

## Available Models

### NV-CLIP
| Feature | Description |
|---|---|
| **Type** | Vision-Language Model |
| **Task** | Zero-shot classification, image-text matching |
| **Input** | Images + text labels/descriptions |
| **Output** | Similarity scores, classification labels |
| **Deployment** | NIM container |

### NV-DINOv2
| Feature | Description |
|---|---|
| **Type** | Self-supervised Vision Transformer |
| **Task** | Few-shot classification, feature extraction |
| **Input** | Images + few labeled examples |
| **Output** | Visual features, classification |

### Vision-Language Models (VLMs) via NIM for LLMs
These are multimodal LLMs that accept both image and text input:

| Model | Provider | Description |
|---|---|---|
| Llama 4 (multimodal) | Meta | Text + image understanding |
| Qwen2-VL | Alibaba | Vision-language understanding |
| Qwen3.5-VL | Alibaba | Latest vision-language model |
| Gemma 3n | Google | Lightweight multimodal |

### NeMo Vision Foundation Models (VFM)
Training and fine-tuning vision models within NeMo Framework:
- Image classification
- Object detection
- Semantic segmentation
- Custom vision model training

## Use Cases

| Task | Recommended Model |
|---|---|
| Zero-shot image classification | NV-CLIP |
| Image-text search | NV-CLIP |
| Few-shot visual classification | NV-DINOv2 |
| Image understanding (questions) | Llama 4, Qwen2-VL |
| Document understanding | Qwen2-VL, Qwen3.5-VL |
| Custom vision tasks | NeMo VFM |
