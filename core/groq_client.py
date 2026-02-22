import io
import logging
import struct
import time
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqKeyPool:
    """Round-robin API key rotation (revolver-style)."""

    def __init__(self, keys: Optional[List[str]] = None):
        self.keys: List[str] = list(keys) if keys else []
        self._index = 0

    def add_key(self, key: str):
        key = key.strip()
        if key and key not in self.keys:
            self.keys.append(key)

    def remove_key(self, key: str):
        key = key.strip()
        if key in self.keys:
            self.keys.remove(key)
            if self._index >= len(self.keys):
                self._index = 0

    def get_next(self) -> Optional[str]:
        if not self.keys:
            return None
        key = self.keys[self._index % len(self.keys)]
        self._index = (self._index + 1) % len(self.keys)
        return key

    @property
    def count(self) -> int:
        return len(self.keys)

    def get_masked_keys(self) -> List[str]:
        """Return keys with middle part masked for display."""
        result = []
        for k in self.keys:
            if len(k) > 12:
                result.append(k[:8] + "..." + k[-4:])
            else:
                result.append("***")
        return result


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy audio array to WAV bytes in memory."""
    if audio.dtype != np.int16:
        if np.issubdtype(audio.dtype, np.floating):
            audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        else:
            audio = audio.astype(np.int16)

    if audio.ndim > 1:
        audio = audio.squeeze()

    num_channels = 1
    sample_width = 2  # int16
    data = audio.tobytes()
    data_size = len(data)

    buf = io.BytesIO()
    # WAV header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 1))   # PCM
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * num_channels * sample_width))
    buf.write(struct.pack("<H", num_channels * sample_width))
    buf.write(struct.pack("<H", sample_width * 8))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(data)
    return buf.getvalue()


class GroqClient:
    """Groq API client with key rotation.

    Uses the OpenAI-compatible API for both STT (Whisper) and text
    processing (Llama 4).
    """

    def __init__(self, key_pool: GroqKeyPool):
        self.key_pool = key_pool

    def _get_client(self):
        """Get an OpenAI client configured for Groq with the next key."""
        from openai import OpenAI

        key = self.key_pool.get_next()
        if not key:
            raise RuntimeError("No Groq API keys configured")
        return OpenAI(api_key=key, base_url=GROQ_BASE_URL)

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        model: str = "whisper-large-v3",
        language: str = "ru",
    ) -> str:
        """Transcribe audio using Groq Whisper API."""
        wav_bytes = _audio_to_wav_bytes(audio, sample_rate)
        client = self._get_client()

        try:
            response = client.audio.transcriptions.create(
                model=model,
                file=("audio.wav", wav_bytes, "audio/wav"),
                language=language,
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str.lower():
                logger.warning("Rate limit hit, trying next key...")
                return self._retry_transcribe(audio, sample_rate, model, language)
            raise

    def _retry_transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        model: str,
        language: str,
    ) -> str:
        """Retry with the next key in the pool."""
        if self.key_pool.count < 2:
            raise RuntimeError("Rate limit hit and no other keys available")

        wav_bytes = _audio_to_wav_bytes(audio, sample_rate)
        client = self._get_client()
        response = client.audio.transcriptions.create(
            model=model,
            file=("audio.wav", wav_bytes, "audio/wav"),
            language=language,
        )
        return response.text

    def process_text(
        self,
        text: str,
        system_prompt: str,
        model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    ) -> str:
        """Process text with Groq LLM (punctuation, translation, etc.)."""
        client = self._get_client()

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
                max_tokens=4096,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str.lower():
                logger.warning("Rate limit hit on text processing, trying next key...")
                return self._retry_process_text(text, system_prompt, model)
            raise

    def _retry_process_text(
        self,
        text: str,
        system_prompt: str,
        model: str,
    ) -> str:
        if self.key_pool.count < 2:
            raise RuntimeError("Rate limit hit and no other keys available")
        client = self._get_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        return response.choices[0].message.content.strip()

    def add_punctuation(
        self,
        text: str,
        prompt: str = "Добавь пунктуацию к следующему русскому тексту. Верни только исправленный текст без пояснений.",
        model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    ) -> str:
        return self.process_text(text, prompt, model)

    def translate(
        self,
        text: str,
        target_language: str = "English",
        prompt: str = "Переведи следующий текст на {language}. Верни только перевод без пояснений.",
        model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    ) -> str:
        system_prompt = prompt.replace("{language}", target_language)
        return self.process_text(text, system_prompt, model)
