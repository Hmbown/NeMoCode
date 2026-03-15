# Speech Models — TTS (Text-to-Speech)

> Models for converting text to natural speech.

## Canonical URLs

| Resource | URL |
|---|---|
| NeMo TTS Docs | `https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/intro.html` |
| Speech NIMs | `https://docs.nvidia.com/nim/speech/latest/` |

## Model Families

### Magpie-TTS (Latest)

| Feature | Description |
|---|---|
| **Architecture** | End-to-end neural TTS |
| **Training** | Preference optimization (DPO/GRPO) for natural output |
| **Longform** | Optimized for long-form speech synthesis |
| **Quality** | Highest quality NVIDIA TTS model |

### Pipeline Models (Mel + Vocoder)

TTS traditionally uses a two-stage pipeline:

**Stage 1: Text → Mel Spectrogram**
| Model | Description |
|---|---|
| FastPitch | Parallel text-to-mel with pitch control |
| Tacotron2 | Autoregressive text-to-mel |
| RAD-TTS | Flow-based with style transfer |

**Stage 2: Mel Spectrogram → Audio**
| Model | Description |
|---|---|
| HiFi-GAN | High-fidelity GAN vocoder |
| UnivNet | Universal vocoder |
| WaveGlow | Flow-based vocoder |

### End-to-End Models
| Model | Description |
|---|---|
| VITS | Variational inference end-to-end TTS |
| Magpie-TTS | Latest end-to-end with preference optimization |

### Audio Enhancement
| Model | Description |
|---|---|
| Audio Enhancer | Post-processing to improve audio quality |
| Audio Codec | Neural audio compression/decompression |

## Features

- **Multiple Voices**: Select from various voice profiles
- **Emotional Styles**: Happy, sad, angry, neutral, and more
- **Voice Cloning**: Clone a voice from reference audio (Speech NIM)
- **Pitch Control**: Fine-grained control over prosody
- **Speed Control**: Adjustable speaking rate
- **SSML Support**: Speech Synthesis Markup Language for detailed control
- **Multi-Language**: Support for multiple languages with text normalization
- **Batch Synthesis**: High-throughput batch processing

## Text Processing Pipeline

1. **Text Normalization** — Convert numbers, abbreviations, symbols to words
2. **Grapheme-to-Phoneme (G2P)** — Convert text to phoneme sequences
3. **Mel Generation** — Produce mel spectrogram from phonemes
4. **Vocoding** — Convert mel spectrogram to audio waveform

## Deployment

```bash
# NIM deployment (TTS with voice cloning)
docker run --gpus all -p 8000:8000 \
  nvcr.io/nim/nvidia/tts:latest
```

## NIM TTS API

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia-tts",
    "input": "Hello, this is a test of NVIDIA text to speech.",
    "voice": "default"
  }' --output speech.wav
```
