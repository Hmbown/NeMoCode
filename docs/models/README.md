# NVIDIA AI Models

> Complete catalog of NVIDIA model families across all modalities.

## Model Families

| Family | Doc | Modality | Description |
|---|---|---|---|
| [Nemotron](nemotron.md) | `docs/models/nemotron.md` | Text | NVIDIA's flagship LLM family |
| [Speech - ASR](speech-asr.md) | `docs/models/speech-asr.md` | Audio→Text | Automatic speech recognition |
| [Speech - TTS](speech-tts.md) | `docs/models/speech-tts.md` | Text→Audio | Text-to-speech synthesis |
| [Embedding](embedding.md) | `docs/models/embedding.md` | Text→Vector | Text and multimodal embeddings |
| [Vision](vision.md) | `docs/models/vision.md` | Image | Classification, detection, understanding |
| [Video](video.md) | `docs/models/video.md` | Video | World models, video generation |
| [Visual GenAI](visual-genai.md) | `docs/models/visual-genai.md` | Text→Image/3D | Image and 3D generation |
| [Digital Human](digital-human.md) | `docs/models/digital-human.md` | Audio→Video | Facial animation, audio processing |

## Where to Access Models

| Method | URL | Description |
|---|---|---|
| Cloud API | `https://build.nvidia.com` | Free serverless endpoints |
| NIM Containers | `https://catalog.ngc.nvidia.com` | Self-hosted inference |
| HuggingFace | `https://huggingface.co/nvidia` | Model weights |
| NeMo Framework | `https://github.com/NVIDIA/NeMo` | Training/fine-tuning |

## Model Naming Conventions

NVIDIA models typically follow: `nvidia/<family>-<size>-<variant>`

Examples:
- `nvidia/nemotron-4-340b-instruct` — Nemotron 4, 340B params, instruction-tuned
- `nvidia/parakeet-tdt-0.6b-v2` — Parakeet ASR, TDT decoder, 0.6B params, v2
- `nvidia/nv-embedqa-e5-v5` — Embedding model for QA, E5-based, v5
