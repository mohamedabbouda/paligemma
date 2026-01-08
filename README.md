# Paligemma (minimal inference)

A small, educational PyTorch implementation of a **PaliGemma-style** multimodal model for **image + text → text** generation:
- **SigLIP** vision encoder
- **Gemma**-style text decoder with KV-cache
- A lightweight processor for image preprocessing and prompt formatting

> This repo focuses on *inference*. Training is not included.

## What’s in this repo

- `inference.py` — CLI entrypoint and token-by-token generation loop
- `processing_paligemma.py` — `PaliGemmaProcessor` (image preprocessing + prompt construction)
- `modeling_siglip.py` — SigLIP vision backbone
- `modeling_gemma.py` — Gemma-like decoder + attention + KV cache
- `utils.py` — tokenizer loading + safetensors weight loading helpers

## Requirements

- Python 3.10+ recommended
- PyTorch
- `transformers`, `tokenizers`
- `safetensors`
- `Pillow`
- `fire` (for the CLI)

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
