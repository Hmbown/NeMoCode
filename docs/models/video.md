# Video & World Foundation Models

> Models for video generation, understanding, and physical AI.

## Canonical URLs

| Resource | URL |
|---|---|
| Cosmos NIM | `https://docs.nvidia.com/nim/cosmos/latest/` |
| build.nvidia.com | `https://build.nvidia.com/explore/discover` (search: Cosmos) |

## Cosmos Family

NVIDIA Cosmos is a family of world foundation models designed for understanding and generating the physical world.

### Cosmos Models

| Model | Type | Description |
|---|---|---|
| **Cosmos Video Generation** | Text/Image→Video | Generate video clips from prompts |
| **Cosmos Embed1** | Video/Image→Vector | Embed visual scenes for retrieval/understanding |
| **Cosmos Tokenizer** | Video→Tokens | Tokenize video for downstream model training |

### Video Generation

Generate realistic video from text descriptions or seed images:

```bash
curl -X POST http://localhost:8000/v1/video/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/cosmos-video",
    "prompt": "A robot arm picking up a red block from a table"
  }'
```

### Video Embeddings (Cosmos Embed1)

Create dense vector representations of video content:
- Scene understanding
- Video search and retrieval
- Temporal relationship encoding
- Action recognition embeddings

### Video Tokenization

Convert video into discrete tokens for:
- Training downstream models
- Video compression
- Multimodal model input

## Use Cases

### Robotics & Physical AI
- Generate synthetic training data for robot manipulation
- Simulate physical interactions
- Create diverse scenario variations

### Autonomous Vehicles
- Generate driving scenarios
- Simulate edge cases (adverse weather, rare events)
- Validate perception systems

### Content Creation
- Text-to-video generation
- Video editing and manipulation
- Style transfer across videos

### Video Understanding
- Scene classification
- Action recognition
- Temporal event detection
- Video question answering (with VLMs)

## Related Models

| Model | Section | Relevance |
|---|---|---|
| Visual GenAI (FLUX, SD) | [visual-genai.md](visual-genai.md) | Image generation |
| Digital Human | [digital-human.md](digital-human.md) | Face/body animation |
| NV-CLIP | [vision.md](vision.md) | Image-level understanding |
