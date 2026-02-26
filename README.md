# index-tts-inference

[![CI](https://github.com/nicokim/indextts2-inference/actions/workflows/ci.yml/badge.svg)](https://github.com/nicokim/indextts2-inference/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/indextts2-inference)](https://pypi.org/project/indextts2-inference/)

Minimal pip package for IndexTTS2 inference. Wraps the official [IndexTTS2](https://github.com/index-tts/index-tts) repo, stripped down to only what's needed for inference.

## Install

```bash
pip install indextts2-inference
```

### Optional extras

**SageAttention** (alternative attention backend):
```bash
pip install indextts2-inference[sage-attn]
```

**Flash Attention v2** (acceleration engine with KV cache and CUDA graphs):
```bash
pip install indextts2-inference[flash-attn]
```

**DeepSpeed**:
```bash
pip install indextts2-inference[deepspeed]
```


## Usage

```python
from indextts import IndexTTS2

# Auto-downloads model from HuggingFace
tts = IndexTTS2()

# Or use local/finetuned checkpoints
tts = IndexTTS2(model_dir="/path/to/checkpoints")

# Basic inference
tts.infer(spk_audio_prompt="voice.wav", text="Hello world", output_path="out.wav")
```

### Attention backends

```python
# Default PyTorch SDPA — auto-selects best kernel, no extra deps needed
tts = IndexTTS2()

# SageAttention — may help on Ampere/Hopper GPUs, requires sageattention package
tts = IndexTTS2(attn_backend="sage", use_fp16=True)

# Flash Attention v2 — acceleration engine with paged KV cache and CUDA graphs
tts = IndexTTS2(attn_backend="flash")
```

### Language selection

By default the language is auto-detected between Chinese and English. You can set it explicitly:

```python
tts = IndexTTS2(language="es")
tts.infer(spk_audio_prompt="voice.wav", text="Hola, esto es una prueba.", output_path="out.wav")
```

### Emotion control

There are three ways to control the emotion of the generated speech:

```python
# 1. From a reference audio
tts.infer(
    spk_audio_prompt="speaker.wav",
    text="Some text",
    output_path="out.wav",
    emo_audio_prompt="happy_reference.wav",
    emo_alpha=0.7,
)

# 2. With an explicit emotion vector
#    [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
tts.infer(
    spk_audio_prompt="speaker.wav",
    text="I am very happy!",
    output_path="out.wav",
    emo_vector=[0.8, 0, 0, 0, 0, 0, 0, 0],
)

# 3. Auto-detect emotion from the text itself
tts.infer(
    spk_audio_prompt="speaker.wav",
    text="I am very happy!",
    output_path="out.wav",
    use_emo_text=True,
)
```

### Streaming

```python
for chunk in tts.infer(
    spk_audio_prompt="voice.wav",
    text="Long text to synthesize...",
    output_path="out.wav",
    stream_return=True,
):
    if chunk is not None and hasattr(chunk, "shape"):
        audio_np = chunk.squeeze().cpu().numpy()
```

### Generation parameters

You can tune sampling parameters via kwargs:

```python
tts.infer(
    spk_audio_prompt="voice.wav",
    text="Hello",
    output_path="out.wav",
    temperature=0.6,
    top_k=20,
    top_p=0.8,
    max_mel_tokens=2000,
)
```

## Logging

By default, `index-tts-inference` only shows warnings. To see detailed logs:

```bash
export INDEXTTS_LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING (default)
```

## PyTorch with CUDA

This package lists `torch` and `torchaudio` as dependencies without pinning a specific CUDA version. Install the CUDA variant you need **before** installing this package:

```bash
# Example: PyTorch with CUDA 12.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Then install the package
pip install indextts2-inference
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

This package is a derivative work of [IndexTTS2](https://github.com/index-tts/index-tts) by Bilibili. Any modifications made to the original model in this derivative work are not endorsed, warranted, or guaranteed by the original right-holder of the original model, and the original right-holder disclaims all liability related to this derivative work.

See [LICENSE](LICENSE) and [DISCLAIMER](DISCLAIMER).
