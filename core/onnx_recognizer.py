from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort

try:
    import sentencepiece as spm
except ImportError:
    spm = None

SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 64
MAX_LETTERS_PER_FRAME = 10
PRED_HIDDEN = 320
BLANK_IDX = 33
VOCAB = (
    " ", "а", "б", "в", "г", "д", "е", "ж", "з", "и", "й",
    "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф",
    "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я",
)


def _hz_to_mel(freq_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq_hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


@dataclass
class MelFeatureExtractor:
    sample_rate: int = SAMPLE_RATE
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    win_length: int = WIN_LENGTH
    n_mels: int = N_MELS

    def __post_init__(self) -> None:
        self.window = np.hanning(self.win_length).astype(np.float32)
        self.filter_bank = self._build_mel_filter_bank()

    def _build_mel_filter_bank(self) -> np.ndarray:
        fft_bins = self.n_fft // 2 + 1
        mel_min = _hz_to_mel(np.array([0.0]))[0]
        mel_max = _hz_to_mel(np.array([self.sample_rate / 2.0]))[0]
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = _mel_to_hz(mel_points)
        bin_indices = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        filter_bank = np.zeros((self.n_mels, fft_bins), dtype=np.float32)
        for m in range(1, self.n_mels + 1):
            left, center, right = bin_indices[m - 1 : m + 2]
            left = max(left, 0)
            if right <= left:
                continue
            for k in range(left, center):
                d = center - left
                if d != 0:
                    filter_bank[m - 1, k] = (k - left) / d
            for k in range(center, right):
                d = right - center
                if d != 0:
                    filter_bank[m - 1, k] = (right - k) / d
        return filter_bank

    def _stft(self, audio: np.ndarray) -> np.ndarray:
        pad = self.win_length // 2
        audio_padded = np.pad(audio, (pad, pad), mode="reflect")
        total_length = audio_padded.shape[0]
        step = self.hop_length
        frames: List[np.ndarray] = []
        for start in range(0, total_length - self.win_length + 1, step):
            frame = audio_padded[start : start + self.win_length]
            spectrum = np.fft.rfft(frame * self.window, n=self.n_fft)
            frames.append(spectrum)
        if not frames:
            frame = audio_padded[: self.win_length]
            spectrum = np.fft.rfft(frame * self.window, n=self.n_fft)
            frames.append(spectrum)
        return np.stack(frames, axis=0)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim > 1:
            audio = audio.squeeze()
        if audio.size == 0:
            return np.zeros((self.n_mels, 0), dtype=np.float32)
        audio = np.asarray(audio)
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
        else:
            audio = audio.astype(np.float32)
        stft_matrix = self._stft(audio)
        power = (np.abs(stft_matrix) ** 2).T
        mel_spec = np.dot(self.filter_bank, power)
        mel_spec = np.maximum(mel_spec, 1e-9)
        log_mel = np.log(mel_spec)
        return log_mel.astype(np.float32)


class OnnxGigaAMTranscriber:
    def __init__(
        self,
        model_dir: Path | str = Path("models") / "onnx",
        model_type: str = "v3_rnnt",
        prefer_cuda: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.full_model_name = model_type

        # Parse "v3_rnnt" -> version="v3", type="rnnt"
        if "_" in model_type and model_type.startswith("v"):
            parts = model_type.split("_", 1)
            self.model_version = parts[0]
            self.model_type = parts[1]
        else:
            self.model_version = "v3"
            self.model_type = model_type

        self.feature_extractor = MelFeatureExtractor()

        # Load tokenizer / vocabulary (for e2e models)
        self.sp_processor = None
        self.vocab_list = None
        self.vocab_size = 0

        tokenizer_path = self.model_dir / f"{self.full_model_name}_tokenizer.model"
        vocab_path = self.model_dir / f"{self.full_model_name}_vocab.txt"

        if tokenizer_path.exists():
            if spm is None:
                raise ImportError("sentencepiece is required for this model")
            self.sp_processor = spm.SentencePieceProcessor()
            self.sp_processor.load(str(tokenizer_path))
            self.vocab_size = self.sp_processor.get_piece_size()
        elif vocab_path.exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocab_list = [line.rstrip("\n") for line in f]
            self.vocab_size = len(self.vocab_list)

        self.is_ctc = self.model_type.endswith("ctc")

        providers: List[str] = []
        available = ort.get_available_providers()
        if prefer_cuda:
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            elif "DmlExecutionProvider" in available:
                providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")

        if self.is_ctc:
            ctc_path = self.model_dir / f"{self.full_model_name}.onnx"
            if not ctc_path.exists():
                raise FileNotFoundError(f"ONNX model not found: {ctc_path}")
            self.encoder = ort.InferenceSession(str(ctc_path), providers=providers)
            self.decoder = None
            self.joint = None
        else:
            base = self.model_dir / self.full_model_name
            encoder_path = base.with_name(base.name + "_encoder.onnx")
            decoder_path = base.with_name(base.name + "_decoder.onnx")
            joint_path = base.with_name(base.name + "_joint.onnx")
            for path in (encoder_path, decoder_path, joint_path):
                if not path.exists():
                    raise FileNotFoundError(f"ONNX model not found: {path}")
            self.encoder = ort.InferenceSession(str(encoder_path), providers=providers)
            self.decoder = ort.InferenceSession(str(decoder_path), providers=providers)
            self.joint = ort.InferenceSession(str(joint_path), providers=providers)

        self.providers = self.encoder.get_providers()

        self.encoder_inputs = self.encoder.get_inputs()
        self.encoder_outputs = self.encoder.get_outputs()
        self.decoder_inputs = self.decoder.get_inputs() if self.decoder else []
        self.decoder_outputs = self.decoder.get_outputs() if self.decoder else []
        self.joint_inputs = self.joint.get_inputs() if self.joint else []
        self.joint_outputs = self.joint.get_outputs() if self.joint else []

        if self.is_ctc:
            out_shape = self.encoder_outputs[0].shape
            if out_shape and isinstance(out_shape[-1], int):
                self.blank_id = out_shape[-1] - 1
            elif self.vocab_list:
                self.blank_id = len(self.vocab_list)
            else:
                self.blank_id = BLANK_IDX
        elif self.joint:
            self.blank_id = self.joint_outputs[0].shape[-1] - 1
        else:
            self.blank_id = BLANK_IDX

    def _decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs to text using available vocabulary."""
        if self.sp_processor:
            valid = [t for t in token_ids if t < self.vocab_size]
            return self.sp_processor.decode(valid)
        if self.vocab_list:
            pieces = [self.vocab_list[t] for t in token_ids if t < len(self.vocab_list)]
            text = "".join(pieces)
            return text.replace("\u2581", " ").strip()
        return "".join(VOCAB[t] for t in token_ids if t < len(VOCAB))

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sample_rate} Hz")

        features = self.feature_extractor(audio)
        if features.shape[1] == 0:
            return ""

        features = features[np.newaxis, :, :]
        lengths = np.array([features.shape[-1]], dtype=np.int64)

        enc_inputs = {
            self.encoder_inputs[0].name: features.astype(np.float32),
            self.encoder_inputs[1].name: lengths,
        }
        enc_features = self.encoder.run(
            [out.name for out in self.encoder_outputs], enc_inputs
        )[0]

        if self.is_ctc:
            logits = enc_features.argmax(axis=1)[0]
            tokens: List[int] = []
            prev = self.blank_id
            for tok in logits:
                tok = int(tok)
                if (tok != prev or prev == self.blank_id) and tok != self.blank_id:
                    tokens.append(tok)
                prev = tok
            return self._decode_tokens(tokens)

        # RNNT decoding
        if self.decoder is None or self.joint is None:
            raise RuntimeError("RNNT decoding requires decoder and joint sessions")

        token_ids: List[int] = []
        prev_token: int = 0
        pred_states = [
            np.zeros((1, 1, PRED_HIDDEN), dtype=np.float32),
            np.zeros((1, 1, PRED_HIDDEN), dtype=np.float32),
        ]

        for frame_idx in range(enc_features.shape[-1]):
            emitted = 0
            while emitted < MAX_LETTERS_PER_FRAME:
                decoder_inputs = {
                    self.decoder_inputs[0].name: np.array([[prev_token]], dtype=np.int64),
                    self.decoder_inputs[1].name: pred_states[0],
                    self.decoder_inputs[2].name: pred_states[1],
                }
                decoder_outputs = self.decoder.run(
                    [out.name for out in self.decoder_outputs], decoder_inputs
                )
                pred_logits = decoder_outputs[0]
                candidate_states = [decoder_outputs[1], decoder_outputs[2]]

                joint_inputs = {
                    self.joint_inputs[0].name: enc_features[:, :, [frame_idx]],
                    self.joint_inputs[1].name: np.swapaxes(pred_logits, 1, 2),
                }
                joint_outputs = self.joint.run(
                    [out.name for out in self.joint_outputs], joint_inputs
                )
                token = int(np.argmax(joint_outputs[0], axis=-1).item())

                if token != self.blank_id:
                    prev_token = token
                    pred_states = candidate_states
                    token_ids.append(token)
                    emitted += 1
                else:
                    break

        return self._decode_tokens(token_ids)
