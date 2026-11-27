# Workshop — Model Frameworks and TTS/ASR Pipelines

v0.70.0 introduced the `workshop` module — a new room for building model
frameworks with pre-built templates for TTS, ASR, LLM, and Vision models.

## Quickstart

```python
from hypernix import workshop

# Build a TTS framework
config = workshop.TTSConfig(
    name="my_tts",
    sample_rate=22050,
    n_mels=80,
    n_speakers=1,
)
tts = workshop.TTSEngine(config)
model = tts.build()

# Build an ASR framework
asr_config = workshop.ASRConfig(name="my_asr")
asr = workshop.ASREngine(asr_config)
asr_model = asr.build()

# Direct speech-to-speech pipeline
pipeline = workshop.ASRToTTS(asr_engine=asr, tts_engine=tts)
audio_output = pipeline.process(audio_input)

# Full conversational pipeline (ASR → LLM → TTS)
conv_pipeline = workshop.ASRToLLMToTTS(
    asr_engine=asr,
    llm=oven,  # a CodeOven from old_oven.preheat() — duck-typed via .chat()
    tts_engine=tts,
)
response_audio = conv_pipeline.process(audio_input)
```

## FrameworkConfig

Base configuration dataclass for all frameworks:

```python
@dataclass
class FrameworkConfig:
    name: str = "base_framework"
    version: str = "0.70.0"
    dtype: torch.dtype = torch.float32
    device: str = "cpu"
    quantization: str | None = None
    checkpoint_path: Path | None = None
```

## TTSEngine

Text-to-Speech synthesis framework. Configuration via `TTSConfig`:

```python
@dataclass
class TTSConfig(FrameworkConfig):
    name: str = "tts_framework"
    sample_rate: int = 22050
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_speakers: int = 1
    max_seq_length: int = 500
    use_gst: bool = False  # Global Style Tokens
    use_pitch_predictor: bool = True
```

Key methods:
- `build()` — Construct the TTS model architecture
- `synthesize(text, speaker_id=0)` — Generate audio from text
- `load_pretrained(path)` — Load weights from checkpoint
- `save(path)` — Save model weights

## ASREngine

Automatic Speech Recognition framework. Configuration via `ASRConfig`:

```python
@dataclass
class ASRConfig(FrameworkConfig):
    name: str = "asr_framework"
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    vocab_size: int = 1000
    use_conformer: bool = True
```

Key methods:
- `build()` — Construct the ASR model architecture
- `transcribe(audio)` — Convert audio to text
- `load_pretrained(path)` — Load weights from checkpoint
- `save(path)` — Save model weights

## ASRToTTS — Direct Speech-to-Speech

Pipeline that chains ASR output directly into TTS input:

```python
pipeline = workshop.ASRToTTS(asr_engine=asr, tts_engine=tts)
output_audio = pipeline.process(input_audio)
```

Useful for voice conversion, translation frontends, or accessibility tools.

## ASRToLLMToTTS — Conversational Pipeline

Full three-stage pipeline: speech → text → LLM response → speech:

```python
pipeline = workshop.ASRToLLMToTTS(
    asr_engine=asr,
    llm=oven,
    tts_engine=tts,
)
response_audio = pipeline.process(user_audio)
```

This is the backbone for voice-enabled chatbots and voice assistants.

`llm` is duck-typed, checked in this order: `.generate(prompt, max_new_tokens=, temperature=)`,
then `.chat(messages, max_new_tokens=)`, then a raw `.forward()` + `.tokenizer`
path, then a fallback stub echo if none match. A `CodeOven` (from
`old_oven.preheat()`, as above) satisfies the `.chat()` path.

## Supported Architectures

The workshop module supports the ray0rf1re/nano-nano collection and 30+
additional architectures including:

- **LiquidAI** — LFM2.5 family
- **MiniCPM** — MiniCPM5
- **Gemma** — Gemma 4 family (e4b, e2b, 26b-a4b, 31b)
- **Qwen** — Qwen3.5 series (0.8b, 2b, 4b, 9b, 27b, 35b-a3b, 122b-a10b, 397b-a17b)
- **Microsoft** — Phi-4
- **DeepSeek** — DeepSeek-V2.5, DeepSeek-R1 distills
- **GLM** — GLM-Edge, GLM-MoE
- **OpenAI** — GPT-OSS (20b, 120b)
- **NVIDIA** — Nemotron family
- **Meta** — Llama-3.2 family
- **Mistral AI** — Mistral-Nemo, Mixtral-8x22B

All architectures work with the same `WorkshopFramework` base class, so
swapping models is a one-line config change.

## Integration with Other Modules

- **`pressure_cooker_v3`** — Train TTS/ASR models with quantization-aware training (FP8/Q8/Q6/Q5.5/Q4M)
- **`compute_framework`** — Multi-GPU DDP training for large TTS/ASR models
- **`freezer`** — VRAM management during training
- **`smoke_alarm`** — Training step planning and monitoring
- **`tv` / `tvtop`** — Real-time training dashboard with metrics streaming

## See Also

- [`wiki/Training.md`](Training.md) — General training flows
- [`wiki/Kitchen.md`](Kitchen.md) — pressure_cooker / pressure_cooker_v3 reference
- [`wiki/Quantization.md`](Quantization.md) — QAT with FP8 and k-quants
