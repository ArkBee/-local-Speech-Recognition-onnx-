import asyncio
import logging
import numpy as np
import time
from async_audio_recorder import AsyncAudioRecorder
from async_keyboard_listener import AsyncKeyboardListener
from onnx_transcriber import OnnxGigaAMTranscriber
import keyboard
from pathlib import Path
import ssl
import certifi
import os
import json
from transcription_utils import process_transcription, extract_transcription_text


ssl._create_default_https_context = lambda *args, **kwargs: ssl.create_default_context(
    cafile=certifi.where()
)

# --- Basic Logging Setup ---
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler("app.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DEDUP_INTERVAL_SECONDS = 1.0
last_transcription_text: str | None = None
last_transcription_time: float = 0.0
last_stop_time: float = 0.0
is_processing: bool = False
enable_replacements: bool = True

# --- Settings Persistence ---
SETTINGS_FILE = Path(__file__).with_name('settings.json')

def load_settings():
    global current_model_idx, enable_replacements
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                current_model_idx = settings.get('current_model_idx', 0)
                enable_replacements = settings.get('enable_replacements', True)
                logger.info(f"Loaded settings: model_idx={current_model_idx}, replacements={enable_replacements}")
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")

def save_settings():
    try:
        settings = {
            'current_model_idx': current_model_idx,
            'enable_replacements': enable_replacements
        }
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4)
        # logger.debug("Settings saved.")
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")

# --- GigaAM Model Initialization ---
AVAILABLE_MODELS = ["v3_rnnt", "v3_e2e_rnnt"]
current_model_idx = 0
load_settings() # Load before initializing model

model_name = AVAILABLE_MODELS[current_model_idx]
transcriber: OnnxGigaAMTranscriber = None  # type: ignore

def load_transcriber(name):
    global transcriber, model_name
    try:
        logger.info(f"Loading ONNX GigaAM model '{name}'...")
        transcriber = OnnxGigaAMTranscriber(model_type=name, prefer_cuda=True)
        model_name = name
        logger.info(f"Model '{name}' loaded successfully on {transcriber.providers[0]}.")
    except Exception as e:
        logger.error(f"Failed to load ONNX model '{name}': {e}")
        if transcriber is None:
            exit(1)

load_transcriber(model_name)

# --- Audio Recorder Initialization ---
# Using default samplerate (48000 Hz) as it's common. 
# GigaAM might have specific requirements; adjust if needed.
# Check GigaAM documentation for its expected audio input format (samplerate, channels, dtype).
# For now, let's assume GigaAM's transcribe_tensor can handle raw audio from the recorder
# or we might need to resample/reformat.
# The default dtype 'int16' from AsyncAudioRecorder is common.
recorder = AsyncAudioRecorder(samplerate=16000) # GigaAM typically expects 16000 Hz

# --- Keyboard Listener Callbacks ---
async def handle_start_recording():
    # logger.info("Hotkey pressed - Starting recording...")
    await recorder.start_recording()

async def handle_stop_recording():
    global is_processing, last_stop_time
    now = time.monotonic()
    
    # Debounce: ignore calls within 500ms of the last successful stop
    if now - last_stop_time < 0.5:
        # logger.debug("Debounce: skipping stop_recording")
        return
        
    if is_processing:
        logger.warning("Already processing a transcription, skipping...")
        return
    
    is_processing = True
    last_stop_time = now
    try:
        audio_data_tuple = await recorder.stop_recording()

        if audio_data_tuple:
            audio_np_array, samplerate = audio_data_tuple

            if audio_np_array.size == 0:
                logger.warning("No audio data captured.")
                return

            try:
                logger.info(f"Transcribing audio with {model_name}...")
                start_time = time.perf_counter()
                
                # Pass numpy array directly to transcriber (it handles conversion to float32)
                raw_text = transcriber.transcribe(audio_np_array, samplerate)
                
                duration = time.perf_counter() - start_time
                logger.info(f"Transcription took {duration:.3f}s")

                if raw_text:
                    if enable_replacements:
                        final_transcription = process_transcription(
                            raw_text,
                            fuzzy_numeric=True,
                            collapse_full_repeat=False,
                        )
                    else:
                        final_transcription = raw_text

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
                        logger.info(f"Processed transcription: {final_transcription}")
                else:
                    logger.warning("No text extracted from transcription.")
            except Exception as e:
                logger.error(f"Error during transcription: {e}", exc_info=True)
        else:
            logger.warning("No audio data was recorded or returned.")
    finally:
        is_processing = False

# --- Main Application Logic ---
async def main():
    logger.info("Initializing application...")

    config_path = Path(__file__).with_name('hotkey_config.txt')
    loaded_hotkey = None
    if config_path.exists():
        try:
            loaded_hotkey = AsyncKeyboardListener.load_hotkey_from_file(config_path)
            logger.info("Loaded hotkey from %s: %s", config_path.name, '+'.join(sorted(loaded_hotkey)))
        except Exception as e:
            logger.error("Failed to load hotkey from %s: %s. Falling back to default (ctrl+alt+shift).", config_path, e)

    keyboard_listener = AsyncKeyboardListener(
        start_recording_callback=handle_start_recording,
        stop_recording_callback=handle_stop_recording,
        hotkey=loaded_hotkey if loaded_hotkey else None
    )

    await keyboard_listener.start_listening()

    # Горячая перезагрузка конфигурации (Ctrl+Shift+Alt+L)
    def _reload_hotkey():
        if not config_path.exists():
            logger.warning("Hotkey config file not found for reload: %s", config_path)
            return
        try:
            keys = AsyncKeyboardListener.load_hotkey_from_file(config_path)
            keyboard_listener.update_hotkey(keys)
            logger.info("Reloaded hotkey from %s: %s", config_path.name, '+'.join(keyboard_listener.current_hotkey))
        except Exception as e:
            logger.error("Failed to reload hotkey: %s", e)

    keyboard.add_hotkey('ctrl+shift+alt+l', _reload_hotkey)

    # Хоткей для переключения моделей (Ctrl+Shift+Alt+M)
    def _switch_model():
        global current_model_idx
        current_model_idx = (current_model_idx + 1) % len(AVAILABLE_MODELS)
        new_model = AVAILABLE_MODELS[current_model_idx]
        load_transcriber(new_model)
        save_settings()
        logger.info(f"Model switched to: {new_model}")

    keyboard.add_hotkey('ctrl+shift+alt+m', _switch_model)

    # Хоткей для включения/выключения замен (Ctrl+Shift+Alt+R)
    def _toggle_replacements():
        global enable_replacements
        enable_replacements = not enable_replacements
        save_settings()
        logger.info(f"Replacements {'enabled' if enable_replacements else 'disabled'}")

    keyboard.add_hotkey('ctrl+shift+alt+r', _toggle_replacements)

    logger.info(
        "Keyboard listener started. Hotkey: %s | Switch Model: Ctrl+Shift+Alt+M | Toggle Replacements: Ctrl+Shift+Alt+R | Reload: Ctrl+Shift+Alt+L | Exit: Ctrl+C",
        '+'.join(keyboard_listener.current_hotkey)
    )

    try:
        # Keep the application running indefinitely
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
            await recorder.stop_recording() # Ensure recorder is stopped if app exits while recording
        logger.info("Application shut down.")

if __name__ == "__main__":
    # Ensure an event loop is available for keyboard listener's threadsafe calls
    # if it's started before asyncio.run() in some contexts (not an issue here with main).
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            logger.info("Asyncio loop closed, exiting.")
        else:
            logger.error(f"Runtime error in main: {e}", exc_info=True)
            raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
