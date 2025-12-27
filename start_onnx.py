import asyncio
import logging
from pathlib import Path
import ssl
import time

import certifi
import keyboard
import numpy as np

from async_audio_recorder import AsyncAudioRecorder
from async_keyboard_listener import AsyncKeyboardListener
from onnx_transcriber import OnnxGigaAMTranscriber
from transcription_utils import process_transcription, extract_transcription_text

ssl._create_default_https_context = lambda *args, **kwargs: ssl.create_default_context(
    cafile=certifi.where()
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEDUP_INTERVAL_SECONDS = 1.0
last_transcription_text: str | None = None
last_transcription_time: float = 0.0

try:
    model = OnnxGigaAMTranscriber(model_type="rnnt", model_version="v2", prefer_cuda=True)
    logger.info("ONNX GigaAM transcriber initialized (providers: %s)", ", ".join(model.providers))
except Exception as e:
    logger.error("Failed to initialize ONNX transcriber: %s", e, exc_info=True)
    raise SystemExit(1)

recorder = AsyncAudioRecorder(samplerate=16000)


async def handle_start_recording():
    await recorder.start_recording()


async def handle_stop_recording():
    audio_data_tuple = await recorder.stop_recording()

    if not audio_data_tuple:
        logger.warning("No audio data was recorded or returned.")
        return

    audio_np_array, samplerate = audio_data_tuple

    if audio_np_array.size == 0:
        logger.warning("No audio data captured.")
        return

    if audio_np_array.dtype in (np.float32, np.float64):
        clipped = np.clip(audio_np_array, -1.0, 1.0)
        audio_int16_np = (clipped * np.iinfo(np.int16).max).astype(np.int16)
    elif audio_np_array.dtype == np.int16:
        audio_int16_np = audio_np_array
    else:
        logger.warning("Unexpected audio data type: %s. Converting to int16 for transcription.", audio_np_array.dtype)
        audio_int16_np = audio_np_array.astype(np.int16)

    audio_int16_1d = audio_int16_np.squeeze()
    if audio_int16_1d.ndim == 0:
        audio_int16_1d = np.array([audio_int16_1d.item()], dtype=np.int16)
    else:
        audio_int16_1d = audio_int16_1d.astype(np.int16, copy=False)

    try:
        logger.info("Transcribing audio via ONNX...")
        transcription_result = model.transcribe(audio_int16_1d, samplerate)

        raw_text = extract_transcription_text(transcription_result)
        if raw_text:
            final_transcription = process_transcription(
                raw_text,
                fuzzy_numeric=True,
                collapse_full_repeat=True,
            )
            global last_transcription_text, last_transcription_time
            now = time.monotonic()
            if (
                last_transcription_text == final_transcription
                and now - last_transcription_time < DEDUP_INTERVAL_SECONDS
            ):
                logger.info(
                    "Skipping duplicate transcription within %.2fs window: %s",
                    DEDUP_INTERVAL_SECONDS,
                    final_transcription,
                )
            else:
                keyboard.write(final_transcription)
                last_transcription_text = final_transcription
                last_transcription_time = now
                logger.info("Processed transcription: %s", final_transcription)
        else:
            logger.warning("No text extracted from transcription result: %s", type(transcription_result))
    except Exception as exc:
        logger.error("Error during ONNX transcription: %s", exc, exc_info=True)


async def main():
    logger.info("Initializing ONNX transcription app...")

    config_path = Path(__file__).with_name("hotkey_config.txt")
    loaded_hotkey = None
    if config_path.exists():
        try:
            loaded_hotkey = AsyncKeyboardListener.load_hotkey_from_file(config_path)
            logger.info("Loaded hotkey from %s: %s", config_path.name, "+".join(sorted(loaded_hotkey)))
        except Exception as e:
            logger.error(
                "Failed to load hotkey from %s: %s. Falling back to default (ctrl+alt+shift).",
                config_path,
                e,
            )

    keyboard_listener = AsyncKeyboardListener(
        start_recording_callback=handle_start_recording,
        stop_recording_callback=handle_stop_recording,
        hotkey=loaded_hotkey if loaded_hotkey else None,
    )

    await keyboard_listener.start_listening()

    def _reload_hotkey():
        if not config_path.exists():
            logger.warning("Hotkey config file not found for reload: %s", config_path)
            return
        try:
            keys = AsyncKeyboardListener.load_hotkey_from_file(config_path)
            keyboard_listener.update_hotkey(keys)
            logger.info(
                "Reloaded hotkey from %s: %s",
                config_path.name,
                "+".join(keyboard_listener.current_hotkey),
            )
        except Exception as e:
            logger.error("Failed to reload hotkey: %s", e)

    keyboard.add_hotkey("f6", _reload_hotkey)

    logger.info(
        "Keyboard listener started (ONNX). Hotkey: %s | Reload: F6 | Exit: Ctrl+C",
        "+".join(keyboard_listener.current_hotkey),
    )

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...")
    except asyncio.CancelledError:
        logger.info("Main task cancelled. Shutting down...")
    finally:
        logger.info("Stopping keyboard listener...")
        await keyboard_listener.stop_listening()
        if recorder.is_recording():
            logger.info("Ensuring recording is stopped...")
            await recorder.stop_recording()
        logger.info("Application shut down.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            logger.info("Asyncio loop closed, exiting.")
        else:
            logger.error("Runtime error in main: %s", e, exc_info=True)
            raise
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e, exc_info=True)
        raise
