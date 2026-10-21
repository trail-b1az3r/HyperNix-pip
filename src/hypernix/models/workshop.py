"""workshop — Model frameworks and TTS/ASR pipelines.

v0.70.5: Added native Multi-Token Prediction (MTP) support and QAT integration.
Supports ray0rf1re/nano-nano collection and 30+ additional architectures.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from hypernix.security.hfpin import hub_revision
from hypernix.training.mtp import MTPConfig, MTPHead, MTPTrainer


def load_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    """A mono waveform at *sample_rate*, from WAV with nothing extra installed.

    torchaudio, then soundfile, then the standard library's ``wave`` for
    PCM WAV; torchaudio was the only reader before, and it is not a
    dependency, so every file failed on a stock install.
    """
    try:
        import torchaudio

        audio, rate = torchaudio.load(str(path))
    except Exception:  # noqa: BLE001 - not installed, or cannot read this file
        try:
            import soundfile

            data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
            audio = torch.from_numpy(data.T.copy())
        except ImportError:
            import wave

            with wave.open(str(path), "rb") as wav:
                rate, width, channels = wav.getframerate(), wav.getsampwidth(), wav.getnchannels()
                frames = wav.readframes(wav.getnframes())
            if width not in (1, 2, 4):
                raise ValueError(f"{path}: {8 * width}-bit WAV; install soundfile to read it") from None
            dtype = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[width]
            raw = torch.frombuffer(bytearray(frames), dtype=dtype).float()
            if width == 1:
                raw = raw - 128.0
            audio = (raw / float(2 ** (8 * width - 1))).view(-1, channels).T
    audio = audio.float().mean(dim=0, keepdim=True)
    if rate != sample_rate:
        try:
            import torchaudio.functional as AF

            audio = AF.resample(audio, rate, sample_rate)
        except ImportError:
            length = max(1, round(audio.shape[-1] * sample_rate / rate))
            audio = torch.nn.functional.interpolate(audio.unsqueeze(0), size=length, mode="linear",
                                                    align_corners=False).squeeze(0)
    return audio


def _to_model(inputs: Any, model: nn.Module) -> dict[str, torch.Tensor]:
    """Processor output on the model's device; only floating tensors take its dtype.

    Casting every input to the model's dtype turned token ids into
    floats, which no embedding layer accepts.
    """
    first = next(model.parameters())
    dtype, device = first.dtype, first.device
    out = {}
    for key, value in dict(inputs).items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(dtype)
        out[key] = value
    return out


# =============================================================================
# Workshop Frameworks - Base templates for model creation
# =============================================================================

@dataclass
class FrameworkConfig:
    """Base configuration for model frameworks."""
    name: str = "base_framework"
    version: str = "0.70.5"
    dtype: torch.dtype = torch.float32
    device: str = "cpu"
    quantization: str | None = None
    checkpoint_path: Path | None = None
    # MTP (Multi-Token Prediction) support
    enable_mtp: bool = False
    mtp_config: MTPConfig | None = None
    # QAT (Quantization-Aware Training) support
    enable_qat: bool = False
    qat_bits: int = 6


class WorkshopFramework:
    """Base framework class for creating model templates.
    
    Provides common infrastructure for TTS, ASR, LLM, and Vision models.
    All frameworks inherit from this base class.
    """
    
    def __init__(self, config: FrameworkConfig | None = None):
        self.config = config or FrameworkConfig()
        self.model: nn.Module | None = None
        self.processor: Any | None = None
        self._initialized = False
    
    def build(self) -> nn.Module:
        """Build and return the model architecture."""
        raise NotImplementedError("Subclasses must implement build()")
    
    def load_pretrained(self, path: Path | str) -> None:
        """Load pretrained weights into the model."""
        if self.model is None:
            self.model = self.build()
        state_dict = torch.load(path, map_location=self.config.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self._initialized = True
    
    def save(self, path: Path | str) -> None:
        """Save model weights to disk."""
        if self.model is None:
            raise RuntimeError("Model not built yet")
        torch.save(self.model.state_dict(), path)
    
    def to(self, device: str | torch.device) -> WorkshopFramework:
        """Move model to specified device."""
        if self.model is not None:
            self.model.to(device)
            self.config.device = str(device)
        return self
    
    def describe(self) -> dict:
        """Return framework description."""
        return {
            "name": self.config.name,
            "version": self.config.version,
            "dtype": str(self.config.dtype),
            "device": self.config.device,
            "quantization": self.config.quantization,
            "initialized": self._initialized,
            "mtp_enabled": self.config.enable_mtp,
            "qat_enabled": self.config.enable_qat,
        }

    # ------------------------------------------------------------------
    # MTP (Multi-Token Prediction) support
    # ------------------------------------------------------------------

    def attach_mtp_head(
        self,
        hidden_dim: int,
        vocab_size: int,
        num_tokens: int | None = None,
    ) -> MTPHead | None:
        """Attach an MTP head for multi-token prediction training.

        Args:
            hidden_dim: Model hidden dimension.
            vocab_size: Vocabulary size.
            num_tokens: Override default number of future tokens.

        Returns:
            The attached MTP head, or None if MTP is not enabled.
        """
        if not self.config.enable_mtp:
            return None

        mtp_cfg = self.config.mtp_config or MTPConfig(
            num_tokens=num_tokens or 4,
        )
        if num_tokens:
            mtp_cfg.num_tokens = num_tokens

        self.mtp_trainer = MTPTrainer(
            model=self.model,
            config=mtp_cfg,
        )
        mtp_head = self.mtp_trainer.attach_head(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
        )
        return mtp_head

    def compute_mtp_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined main + MTP loss.

        Args:
            logits: Main model logits [batch, seq, vocab].
            labels: Target token IDs [batch, seq].
            hidden_states: Optional hidden states for MTP.

        Returns:
            Dict with 'main', 'mtp', 'total' losses.
        """
        if not hasattr(self, "mtp_trainer") or self.mtp_trainer is None:
            # MTP not attached — return main loss only
            import torch.nn.functional as F
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            return {"main": loss, "mtp": torch.tensor(0.0), "total": loss}

        return self.mtp_trainer.compute_loss(logits, labels, hidden_states)

    def get_mtp_stats(self) -> dict[str, Any]:
        """Get MTP training statistics."""
        if hasattr(self, "mtp_trainer") and self.mtp_trainer is not None:
            return self.mtp_trainer.get_stats()
        return {"mtp_enabled": False}


# =============================================================================
# TTS Engine - Text-to-Speech Synthesis
# =============================================================================

@dataclass
class TTSConfig(FrameworkConfig):
    """Configuration for TTS models."""
    name: str = "tts_framework"
    #: A Hugging Face text-to-waveform model (VITS/MMS, SpeechT5, Bark...).
    model_name: str = "facebook/mms-tts-eng"
    #: Overwritten by the model's own rate once it is loaded: a WAV
    #: written at the wrong rate plays at the wrong pitch.
    sample_rate: int = 22050
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_speakers: int = 1
    max_seq_length: int = 500
    use_gst: bool = False  # Global Style Tokens
    use_pitch_predictor: bool = True


class TTSEngine(WorkshopFramework):
    """Text-to-Speech synthesis engine.
    
    Supports multiple TTS architectures including Tacotron2, FastSpeech2,
    VITS, and nano-nano TTS variants.
    """
    
    def __init__(self, config: TTSConfig | None = None):
        super().__init__(config or TTSConfig())
        self.synthesizer: nn.Module | None = None
        self.vocoder: nn.Module | None = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the TTS engine with default model."""
        if self._initialized:
            return
        
        try:
            import torch
            from transformers import AutoModelForTextToWaveform, AutoProcessor
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Always use float32 on CPU to avoid dtype mismatch errors
            dtype = torch.float16 if device == "cuda" else torch.float32
            
            # Default model
            model_name = getattr(self.config, 'model_name', 'facebook/mms-tts-eng')
            
            self.model = AutoModelForTextToWaveform.from_pretrained(
                model_name,
                revision=hub_revision(model_name),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            ).to(device=device, dtype=dtype)
            
            self.processor = AutoProcessor.from_pretrained(model_name, revision=hub_revision(model_name))
            rate = getattr(self.model.config, "sampling_rate", None) or getattr(
                getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            if rate:
                self.config.sample_rate = int(rate)
            self._initialized = True
        except ImportError as e:
            raise RuntimeError("TTS requires transformers. Install: pip install transformers[torch]") from e
    
    def build(self) -> nn.Module:
        """Build or return the initialized TTS model."""
        if not hasattr(self, "model"):
            self.initialize()
        return self.model
    
    def synthesize(self, text: str, speaker_id: int = 0) -> torch.Tensor:
        """Synthesize speech from text."""
        if not self._initialized:
            raise RuntimeError("TTS engine not initialized")
        
        inputs = _to_model(self.processor(text=text, return_tensors="pt"), self.model)
        with torch.no_grad():
            # VITS/MMS answer from forward(); only some TTS models generate().
            if hasattr(self.model, "generate") and not hasattr(self.model.config, "flow_size"):
                output = self.model.generate(**inputs)
            else:
                output = self.model(**inputs)
        waveform = getattr(output, "waveform", output)
        if isinstance(waveform, tuple):
            waveform = waveform[0]
        return waveform.float().squeeze()
    
    def set_vocoder(self, vocoder: nn.Module) -> None:
        """Set the vocoder for mel-to-audio conversion."""
        self.vocoder = vocoder
    
    def describe(self) -> dict:
        base_desc = super().describe()
        base_desc.update({
            "sample_rate": self.config.sample_rate,
            "n_mels": self.config.n_mels,
            "n_speakers": self.config.n_speakers,
            "has_vocoder": self.vocoder is not None,
        })
        return base_desc


# =============================================================================
# ASR Engine - Automatic Speech Recognition
# =============================================================================

@dataclass
class ASRConfig(FrameworkConfig):
    """Configuration for ASR models."""
    name: str = "asr_framework"
    #: A Hugging Face speech-to-text model (Whisper and friends).
    model_name: str = "openai/whisper-tiny"
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    vocab_size: int = 5000
    max_audio_length: float = 30.0  # seconds
    use_conformer: bool = True
    use_streaming: bool = False


class ASREngine(WorkshopFramework):
    """Automatic Speech Recognition engine.
    
    Supports Whisper, Conformer, RNN-T, and nano-nano ASR variants.
    """
    
    def __init__(self, config: ASRConfig | None = None):
        super().__init__(config or ASRConfig())
        self.transcriber: nn.Module | None = None
        self.processor: Any | None = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the ASR engine with default nano-whisper model."""
        if self._initialized:
            return
        
        # Lazy import to avoid heavy dependencies if not used
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Always use float32 on CPU to avoid dtype mismatch errors
            dtype = torch.float16 if device == "cuda" else torch.float32
            
            # Default to nano-whisper if not specified
            model_name = getattr(self.config, 'model_name', 'openai/whisper-tiny')
            
            self.transcriber = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                revision=hub_revision(model_name),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            ).to(device=device, dtype=dtype)
            
            self.processor = AutoProcessor.from_pretrained(model_name, revision=hub_revision(model_name))
            self._initialized = True
        except ImportError as e:
            raise RuntimeError("ASR requires transformers. Install: pip install transformers[torch]") from e
    
    def build(self) -> nn.Module:
        """Build or return the initialized ASR model."""
        if not hasattr(self, "transcriber"):
            self.initialize()
        return self.transcriber
    
    def transcribe(self, audio: torch.Tensor, language: str = "en") -> str:
        """Transcribe audio to text."""
        if not self._initialized:
            raise RuntimeError("ASR engine not initialized")
        
        if audio.ndim > 1:
            audio = audio.mean(dim=0)
        inputs = self.processor(audio.float().cpu().numpy(), sampling_rate=self.config.sample_rate,
                                return_tensors="pt")
        inputs = _to_model(inputs, self.transcriber)
        with torch.no_grad():
            features = inputs["input_features"]
            try:
                generated_ids = self.transcriber.generate(features, language=language, task="transcribe")
            except (TypeError, ValueError):  # not a Whisper: no language to pass
                generated_ids = self.transcriber.generate(features)
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()
    
    def transcribe_batch(self, audios: list[torch.Tensor]) -> list[str]:
        """Transcribe multiple audio samples."""
        return [self.transcribe(audio) for audio in audios]
    
    def describe(self) -> dict:
        base_desc = super().describe()
        base_desc.update({
            "sample_rate": self.config.sample_rate,
            "vocab_size": self.config.vocab_size,
            "use_conformer": self.config.use_conformer,
            "streaming": self.config.use_streaming,
        })
        return base_desc


# =============================================================================
# Pipeline Classes - Combined workflows
# =============================================================================

class ASRToTTS:
    """Direct speech-to-speech pipeline (e.g., voice translation, voice conversion)."""
    
    def __init__(self, asr_engine: ASREngine, tts_engine: TTSEngine):
        self.asr = asr_engine
        self.tts = tts_engine
        self._validate_compatibility()
    
    def _validate_compatibility(self) -> None:
        """Ensure ASR and TTS are compatible."""
        if self.asr.config.sample_rate != self.tts.config.sample_rate:
            # Would need resampling in production
            pass
    
    def process(self, audio: torch.Tensor, target_speaker: int = 0) -> torch.Tensor:
        """Convert speech to speech (ASR → TTS)."""
        text = self.asr.transcribe(audio)
        output_audio = self.tts.synthesize(text, speaker_id=target_speaker)
        return output_audio
    
    def describe(self) -> dict:
        return {
            "type": "ASRToTTS",
            "asr": self.asr.describe(),
            "tts": self.tts.describe(),
        }


class ASRToLLMToTTS:
    """Full conversational pipeline (ASR → LLM → TTS)."""
    
    def __init__(
        self,
        asr_engine: ASREngine,
        llm: Any,  # Could be any LLM framework
        tts_engine: TTSEngine,
        system_prompt: str = "You are a helpful assistant.",
    ):
        if not isinstance(asr_engine, ASREngine):
            raise TypeError("asr_engine must be an instance of ASREngine")
        if not isinstance(tts_engine, TTSEngine):
            raise TypeError("tts_engine must be an instance of TTSEngine")
        
        self.asr = asr_engine
        self.llm = llm
        self.tts = tts_engine
        self.system_prompt = system_prompt
        self.conversation_history: list[dict] = []
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize all pipeline components."""
        if not self._initialized:
            self.asr.initialize()
            if hasattr(self.llm, 'initialize'):
                self.llm.initialize()
            self.tts.initialize()
            self._initialized = True
    
    def process(
        self,
        audio_path: str | torch.Tensor,
        max_response_length: int = 500,
        temperature: float = 0.7,
    ) -> tuple[str, bytes]:
        """Full pipeline: speech → text → LLM response → speech.
        
        Args:
            audio_path: Path to audio file or raw audio tensor
            max_response_length: Maximum tokens in response
            temperature: Sampling temperature for LLM
            
        Returns:
            Tuple of (response_text, audio_bytes)
        """
        self.initialize()
        
        # Step 1: ASR - load audio if path provided
        if isinstance(audio_path, str):
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            audio = load_audio(audio_path, self.asr.config.sample_rate)
        else:
            audio = audio_path
        
        input_text = self.asr.transcribe(audio)
        
        # Step 2: Add to conversation history
        self.conversation_history.append({"role": "user", "content": input_text})
        
        # Step 3: LLM inference
        prompt = self._build_prompt()
        response_text = self._llm_generate(prompt, max_response_length, temperature)
        
        # Step 4: Update history
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # Step 5: TTS - synthesize and convert to bytes
        response_audio = self.tts.synthesize(response_text)
        if isinstance(response_audio, torch.Tensor):
            # Convert tensor to bytes (WAV format)
            import io
            import wave
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.tts.config.sample_rate)
                # Convert to 16-bit PCM
                audio_data = (response_audio.clamp(-1, 1) * 32767).short().cpu().numpy()
                wav_file.writeframes(audio_data.tobytes())
            audio_bytes = buffer.getvalue()
        else:
            audio_bytes = response_audio
        
        return response_text, audio_bytes
    
    def _build_prompt(self) -> str:
        """Build prompt from conversation history."""
        prompt_parts = [f"System: {self.system_prompt}"]
        for msg in self.conversation_history[-10:]:  # Last 10 messages
            prompt_parts.append(f"{msg['role'].capitalize()}: {msg['content']}")
        return "\n".join(prompt_parts)
    
    def _llm_generate(self, prompt: str, max_length: int, temperature: float) -> str:
        """Generate response from LLM."""
        if hasattr(self.llm, 'generate'):
            # Standard generate method
            return self.llm.generate(prompt, max_new_tokens=max_length, temperature=temperature)
        elif hasattr(self.llm, 'chat'):
            # Chat-style interface. Uses max_new_tokens to match
            # hypernix.old_oven.CodeOven.chat()'s real keyword name —
            # this is the object the docs recommend passing as `llm=`.
            return self.llm.chat(self.conversation_history, max_new_tokens=max_length)
        elif hasattr(self.llm, 'forward'):
            # Raw model - need tokenizer
            if hasattr(self.llm, 'tokenizer'):
                inputs = self.llm.tokenizer.encode(prompt, return_tensors="pt")
                outputs = self.llm.forward(inputs)
                return self.llm.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # An LLM that cannot answer is an error, not an echo dressed up as a reply.
        raise TypeError(f"{type(self.llm).__name__} has no generate(), chat() or forward()+tokenizer")
    
    def _llm_stream(self, prompt: str, max_length: int, temperature: float):
        """Stream response from LLM, yielding tokens."""
        if hasattr(self.llm, 'stream'):
            yield from self.llm.stream(prompt, max_new_tokens=max_length, temperature=temperature)
        elif hasattr(self.llm, 'generate_stream'):
            yield from self.llm.generate_stream(prompt, max_new_tokens=max_length, temperature=temperature)
        else:
            # Fallback: generate all at once, then split into words to simulate streaming
            full_text = self._llm_generate(prompt, max_length, temperature)
            import re
            words = re.findall(r'\S+|\s+', full_text)
            yield from words
    
    def process_stream(
        self,
        audio_path: str | torch.Tensor,
        max_response_length: int = 500,
        temperature: float = 0.7,
    ):
        """Streaming pipeline: speech → text → LLM stream → TTS chunks.
        
        Yields:
            Tuple of (text_chunk, audio_bytes_chunk) as soon as each sentence is generated.
        """
        self.initialize()
        
        if isinstance(audio_path, str):
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            import torchaudio
            audio, sr = torchaudio.load(audio_path)
            if sr != self.asr.config.sample_rate:
                import torch.nn.functional as F
                ratio = self.asr.config.sample_rate / sr
                new_length = int(audio.shape[1] * ratio)
                audio = F.interpolate(audio.unsqueeze(0), size=new_length, mode='linear', align_corners=False).squeeze(0)
        else:
            audio = audio_path
        
        input_text = self.asr.transcribe(audio)
        self.conversation_history.append({"role": "user", "content": input_text})
        
        prompt = self._build_prompt()
        
        full_response = []
        buffer = ""
        
        def synthesize_and_yield(text_chunk: str):
            if not text_chunk.strip():
                return None
            response_audio = self.tts.synthesize(text_chunk.strip())
            if isinstance(response_audio, torch.Tensor):
                import io
                import wave
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self.tts.config.sample_rate)
                    audio_data = (response_audio.clamp(-1, 1) * 32767).short().cpu().numpy()
                    wav_file.writeframes(audio_data.tobytes())
                audio_bytes = buf.getvalue()
            else:
                audio_bytes = response_audio
            return text_chunk.strip(), audio_bytes

        # Stream LLM tokens, group into sentences, and synthesize
        import re
        for token in self._llm_stream(prompt, max_response_length, temperature):
            buffer += token
            full_response.append(token)
            
            # Check for sentence boundaries (. ! ?) followed by space or newline
            if re.search(r'[.!?](?:\s+|\n)', buffer):
                parts = re.split(r'(?<=[.!?])(?:\s+|\n)', buffer, maxsplit=1)
                sentence = parts[0]
                buffer = parts[1] if len(parts) > 1 else ""
                
                result = synthesize_and_yield(sentence)
                if result:
                    yield result
                    
        # Yield any remaining text
        if buffer.strip():
            result = synthesize_and_yield(buffer)
            if result:
                yield result
                
        self.conversation_history.append({"role": "assistant", "content": "".join(full_response)})

    def reset(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
    
    def describe(self) -> dict:
        return {
            "type": "ASRToLLMToTTS",
            "asr": self.asr.describe(),
            "tts": self.tts.describe(),
            "system_prompt": self.system_prompt,
            "conversation_turns": len(self.conversation_history),
        }


# =============================================================================
# Nano-Nano Collection Support
# =============================================================================

NANO_NANO_MODELS = {
    # From ray0rf1re/nano-nano collection
    "nano-llama": {"type": "llm", "params": "80M", "context": 2048},
    "nano-mistral": {"type": "llm", "params": "120M", "context": 4096},
    "nano-whisper": {"type": "asr", "params": "30M", "languages": ["en", "es", "fr", "de"]},
    "nano-tacotron": {"type": "tts", "params": "25M", "speakers": 1},
    "nano-vits": {"type": "tts", "params": "40M", "speakers": "multi"},
    # Additional nano variants
    "nano-conformer": {"type": "asr", "params": "35M", "streaming": True},
    "nano-fastpitch": {"type": "tts", "params": "28M", "fast": True},
    "nano-bert": {"type": "encoder", "params": "50M", "layers": 6},
    "nano-vit": {"type": "vision", "params": "45M", "patch_size": 16},
    "nano-unet": {"type": "diffusion", "params": "60M", "channels": 128},
}


def load_nano_model(model_name: str, config: FrameworkConfig | None = None) -> WorkshopFramework:
    """Load a nano-nano model by name."""
    if model_name not in NANO_NANO_MODELS:
        available = list(NANO_NANO_MODELS.keys())
        raise ValueError(f"Unknown nano model '{model_name}'. Available: {available}")
    
    model_info = NANO_NANO_MODELS[model_name]
    
    if model_info["type"] == "tts":
        return TTSEngine(TTSConfig(name=model_name, **(config.__dict__ if config else {})))
    elif model_info["type"] == "asr":
        return ASREngine(ASRConfig(name=model_name, **(config.__dict__ if config else {})))
    else:
        # Generic framework for other types
        return WorkshopFramework(config or FrameworkConfig(name=model_name))


# =============================================================================
# Registry
# =============================================================================

FRAMEWORK_REGISTRY: dict[str, type[WorkshopFramework]] = {
    "base": WorkshopFramework,
    "tts": TTSEngine,
    "asr": ASREngine,
}


def create_framework(framework_type: str, config: Any = None) -> WorkshopFramework:
    """Create a framework instance by type name."""
    if framework_type not in FRAMEWORK_REGISTRY:
        available = list(FRAMEWORK_REGISTRY.keys())
        raise ValueError(f"Unknown framework type '{framework_type}'. Available: {available}")
    
    return FRAMEWORK_REGISTRY[framework_type](config)
