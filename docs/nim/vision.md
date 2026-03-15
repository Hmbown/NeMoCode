# NIM for Vision & Multimodal

> Vision-language, video generation, image generation, and digital human NIMs.

## Canonical URLs

| Resource | URL |
|---|---|
| NV-CLIP | `https://docs.nvidia.com/nim/nvclip/latest/` |
| Cosmos | `https://docs.nvidia.com/nim/cosmos/latest/` |
| Visual GenAI | `https://docs.nvidia.com/nim/visual-genai/latest/` |

## NV-CLIP

Vision-language model for zero-shot image classification and image-text matching.

### Capabilities
- Zero-shot image classification (no fine-tuning needed)
- Image-text similarity scoring
- Visual search and retrieval
- Few-shot classification

### API
```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "nvidia/nv-clip", "input": {"image": "<base64>"}}'
```

## Cosmos (World Foundation Models)

Video generation and understanding models for physical AI.

### Models
| Model | Description |
|---|---|
| Cosmos Video Generation | Generate videos from text or image prompts |
| Cosmos Embed1 | Video/image embeddings for world understanding |
| Cosmos Tokenizer | Video tokenization for downstream tasks |

### Use Cases
- Robotics simulation data generation
- Autonomous vehicle scenario generation
- Physical AI training data
- Video understanding and retrieval

## Visual Generative AI

Image and 3D generation models.

### Available Models
| Model | Type | Description |
|---|---|---|
| FLUX.1-dev | Image | High-quality text-to-image |
| FLUX.1-schnell | Image | Fast text-to-image |
| FLUX.1-Kontext-dev | Image | Context-aware image editing |
| FLUX.2-klein | Image | Compact FLUX variant |
| Stable Diffusion 3.5 Large | Image | Stability AI's latest |
| TRELLIS | 3D | Text/image to 3D model generation |

### API Pattern
```bash
curl -X POST http://localhost:8000/v1/images/generations \
  -d '{"model": "black-forest-labs/flux.1-dev", "prompt": "A sunset over mountains"}'
```

## Digital Human / Maxine

| Model | Description |
|---|---|
| Audio2Face-3D | Generate 3D facial animation from audio |
| Audio2Face-2D | Generate 2D facial animation from audio |
| Eye Contact | Real-time gaze correction for video calls |
| Studio Voice | Audio enhancement and noise removal |

## NV-DINOv2

Few-shot visual classification using vision transformer features. Classify images with minimal labeled examples.
