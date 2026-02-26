# index-tts-inference

Minimal pip package for IndexTTS2 inference. Wraps the official [IndexTTS2](https://github.com/index-tts/index-tts) repo, stripped down to only what's needed for inference.

## Install

```bash
pip install index-tts-inference
```

### Optional extras

**Flash Attention** (faster inference on CUDA):
```bash
pip install index-tts-inference[flash-attn]
```

**DeepSpeed**:
```bash
pip install index-tts-inference[deepspeed]
```

> Note: Both `flash-attn` and `deepspeed` require build from source and need CUDA toolkit installed.

## Usage

```python
from indextts import IndexTTS2

# Auto-downloads model from HuggingFace
tts = IndexTTS2()

# Or use local/finetuned checkpoints
tts = IndexTTS2(model_dir="/path/to/checkpoints")

# Basic inference
tts.infer(spk_audio_prompt="voice.wav", text="Hello world", output_path="out.wav")

# Streaming
for chunk in tts.infer(spk_audio_prompt="voice.wav", text="Hello", output_path="out.wav", stream_return=True):
    if chunk is not None and hasattr(chunk, "shape"):
        audio_np = chunk.squeeze().cpu().numpy()
```

## PyTorch with CUDA

This package lists `torch` and `torchaudio` as dependencies without pinning a specific CUDA version. Install the CUDA variant you need **before** installing this package:

```bash
# Example: PyTorch with CUDA 12.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Then install the package
pip install index-tts-inference
```

Or with uv:
```toml
# pyproject.toml of your project
[tool.uv.sources]
torch = [{ index = "pytorch-cuda", marker = "sys_platform == 'linux'" }]
torchaudio = [{ index = "pytorch-cuda", marker = "sys_platform == 'linux'" }]

[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

## License

See [LICENSE](LICENSE) and [DISCLAIMER](DISCLAIMER).
