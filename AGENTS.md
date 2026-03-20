# Repository Agent Rules

- Do not introduce, recommend, or switch NeMoCode coding backends to Llama-based models unless the user explicitly asks for one.
- Prefer native NVIDIA Nemotron models over Llama-Nemotron variants when selecting or updating default model choices.
- For Nano 4B choices, prefer the native Nemotron 3 Nano 4B family and verify the exact backend-specific model ID against official NVIDIA sources before changing code.
- As of March 16, 2026, the official self-hosted Hugging Face checkpoints are `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` and `nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8`.
