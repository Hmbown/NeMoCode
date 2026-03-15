# Speech Models — ASR (Automatic Speech Recognition)

> Models for converting speech to text.

## Canonical URLs

| Resource | URL |
|---|---|
| NeMo ASR Docs | `https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html` |
| Speech NIMs | `https://docs.nvidia.com/nim/speech/latest/` |
| HuggingFace | `https://huggingface.co/nvidia` (search "parakeet" or "canary") |

## Model Families

### Parakeet (English, High-Accuracy)

| Model | Size | Decoder | Notes |
|---|---|---|---|
| **Parakeet-TDT-0.6B-V2** | 0.6B | TDT | #1 on HuggingFace OpenASR leaderboard |
| **Parakeet-CTC-1.1B** | 1.1B | CTC | CTC decoder, good for streaming |
| **Parakeet-RNNT-1.1B** | 1.1B | RNN-T | RNN-Transducer decoder |
| **Parakeet-TDT-1.1B** | 1.1B | TDT | Token-and-Duration Transducer |

### Canary (Multilingual)

| Model | Languages | Description |
|---|---|---|
| **Canary-1B** | EN, ES, DE, FR | Multilingual with automatic punctuation & capitalization |

### Conformer (General Purpose)

| Model | Description |
|---|---|
| **Conformer-CTC** | CTC decoder, available in multiple sizes and languages |
| **Conformer-Transducer** | RNN-T decoder variant |
| **FastConformer** | Optimized Conformer architecture |

### Whisper (OpenAI, NVIDIA-optimized)

| Model | Languages | Description |
|---|---|---|
| **Whisper Large v3** | 100+ | OpenAI's Whisper optimized with TensorRT |

### Nemotron ASR

| Model | Description |
|---|---|
| **Nemotron ASR Streaming** | Optimized for real-time streaming ASR |

## Encoder Architectures

| Architecture | Description |
|---|---|
| **Conformer** | Convolution + Transformer hybrid — best quality |
| **FastConformer** | Optimized Conformer with faster inference |
| **Citrinet** | CNN-based — lighter weight |

## Decoder Types

| Decoder | Streaming | Latency | Quality |
|---|---|---|---|
| **CTC** | Yes (natural) | Lowest | Good |
| **RNN-T** | Yes | Low | Better |
| **TDT** | Yes | Low | Best (Parakeet) |
| **Attention (AED)** | No (batch) | Higher | Best (Canary) |

## Features

- **Streaming**: Cache-aware and buffered streaming modes
- **14+ Languages**: English, Spanish, German, French, Mandarin, Japanese, Korean, and more
- **Punctuation & Capitalization**: Automatic in Canary models
- **Word Timestamps**: Token-level timing information
- **Speaker Diarization**: Identify who spoke when
- **Custom Vocabulary**: Hot-word boosting for domain-specific terms

## Deployment

```bash
# NIM deployment
docker run --gpus all -p 50051:50051 \
  nvcr.io/nim/nvidia/parakeet-tdt-0.6b-v2:latest

# NeMo Framework (training/fine-tuning)
python examples/asr/speech_to_text.py \
  model.pretrained_model_name="nvidia/parakeet-tdt-0.6b-v2"
```
