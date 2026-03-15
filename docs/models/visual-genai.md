# Visual Generative AI Models

> Image generation and 3D generation models.

## Canonical URL

`https://docs.nvidia.com/nim/visual-genai/latest/`

## Image Generation Models

### FLUX Family (Black Forest Labs)

| Model | Speed | Quality | Description |
|---|---|---|---|
| **FLUX.1-dev** | Medium | Highest | Development/research, best quality |
| **FLUX.1-schnell** | Fast | High | Fast generation, production use |
| **FLUX.1-Kontext-dev** | Medium | High | Context-aware image editing |
| **FLUX.2-klein** | Fastest | Good | Compact variant for efficiency |

### Stable Diffusion (Stability AI)

| Model | Description |
|---|---|
| **Stable Diffusion 3.5 Large** | Latest SD model, high quality and detail |

## 3D Generation Models

### TRELLIS

| Feature | Description |
|---|---|
| **Input** | Text or image |
| **Output** | 3D model (mesh, texture) |
| **Use Cases** | Game assets, product visualization, prototyping |

## API (Image Generation)

```bash
# Text-to-image
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "black-forest-labs/flux.1-dev",
    "prompt": "A futuristic cityscape at sunset, photorealistic",
    "n": 1,
    "size": "1024x1024"
  }'
```

```bash
# Image editing (FLUX Kontext)
curl -X POST http://localhost:8000/v1/images/edits \
  -H "Content-Type: application/json" \
  -d '{
    "model": "black-forest-labs/flux.1-kontext-dev",
    "prompt": "Change the sky to a starry night",
    "image": "<base64-encoded-image>"
  }'
```

## Deployment

Available as NIM containers:

```bash
# FLUX.1-dev NIM
docker run --gpus all -p 8000:8000 \
  nvcr.io/nim/black-forest-labs/flux.1-dev:latest
```

Also available via cloud API at `build.nvidia.com`.

## Use Cases

| Task | Recommended Model |
|---|---|
| Highest quality image generation | FLUX.1-dev |
| Fast production image generation | FLUX.1-schnell |
| Image editing / inpainting | FLUX.1-Kontext-dev |
| Efficient batch generation | FLUX.2-klein |
| 3D asset generation | TRELLIS |
| General purpose | Stable Diffusion 3.5 Large |
