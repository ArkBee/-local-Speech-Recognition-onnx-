import logging
import threading
from collections import deque
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
BLOCKSIZE = 1024


class AudioRecorder:
    """Synchronous audio recorder using sounddevice callbacks.

    Thread-safe: the sounddevice callback runs in a separate thread and
    appends chunks to a deque. start/stop are called from the main (GUI)
    thread.
    """

    def __init__(
        self,
        samplerate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        dtype: str = DTYPE,
        blocksize: int = BLOCKSIZE,
    ):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.blocksize = blocksize

        self._stream: Optional[sd.InputStream] = None
        self._buffer: deque = deque()
        self._recording = False
        self._lock = threading.Lock()
        self._current_rms: float = 0.0

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            logger.warning("sounddevice status: %s", status)
        if self._recording:
            self._buffer.append(indata.copy())
            self._current_rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))

    def start(self):
        with self._lock:
            if self._recording:
                return
            self._buffer.clear()
            self._recording = True
            self._stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=self.blocksize,
                callback=self._callback,
            )
            self._stream.start()
            logger.info("Recording started (sr=%d)", self.samplerate)

    def stop(self) -> Optional[Tuple[np.ndarray, int]]:
        with self._lock:
            if not self._recording:
                return None
            self._recording = False
            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.error("Error stopping stream: %s", e)
                finally:
                    self._stream = None

            if not self._buffer:
                logger.info("No audio data recorded")
                return None

            data = np.concatenate(list(self._buffer))
            self._buffer.clear()
            logger.info("Recorded %d samples (%.2fs)", len(data), len(data) / self.samplerate)
            return data, self.samplerate

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def current_rms(self) -> float:
        return self._current_rms
