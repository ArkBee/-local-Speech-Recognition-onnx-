import argparse
import logging
import sys
from pathlib import Path

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "onnx"

DEFAULT_HOTKEYS = {
    "raw": ["ctrl", "alt"],
    "punctuation": ["ctrl", "shift"],
    "translate": ["ctrl", "shift", "alt"],
}


def _acquire_singleton():
    """Prevent double launch using a Windows named mutex.

    Returns the mutex handle (must be kept alive) or None on non-Windows.
    Exits the process if another instance is already running.
    """
    if sys.platform != "win32":
        return None

    import ctypes
    kernel32 = ctypes.windll.kernel32

    MUTEX_NAME = "Global\\VoiceRecognition_SingleInstance"
    ERROR_ALREADY_EXISTS = 183

    handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
    if kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
        # Check if the existing instance is actually alive (has a window)
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, "Voice Recognition - GigaAM v3 + Groq")
        if hwnd:
            # Real instance running — bring it to front and exit
            SW_RESTORE = 9
            user32.ShowWindow(hwnd, SW_RESTORE)
            user32.SetForegroundWindow(hwnd)
            kernel32.CloseHandle(handle)
            logger.warning("Another instance is already running. Exiting.")
            sys.exit(0)
        else:
            # Zombie process — no window found, proceed with our handle
            logger.warning("Stale mutex detected (no window found), taking over.")

    return handle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimized", action="store_true", help="Start minimized to system tray")
    args = parser.parse_args()

    # Singleton check — prevent double launch
    _mutex = _acquire_singleton()  # noqa: F841 — prevent GC from releasing mutex

    from gui.app import App
    from gui.test_tab import TestTab
    from gui.hotkeys_tab import HotkeysTab
    from gui.groq_tab import GroqTab
    from gui.history_tab import HistoryTab
    from gui.recording_overlay import RecordingOverlay, play_start_sound, play_stop_sound
    from core.audio_recorder import AudioRecorder
    from core.groq_client import GroqClient, GroqKeyPool
    from core.hotkey_manager import MultiHotkeyManager
    from core.model_downloader import check_models_exist, download_models

    # --- GUI ---
    app = App()
    settings = app.settings

    # --- Download models if needed ---
    model_type = settings.get("model", "v3_rnnt")
    if not check_models_exist(MODEL_DIR, model_type):
        app.set_status(f"Скачивание модели {model_type}...")
        app.update()
        logger.info("Downloading ONNX models: %s", model_type)

        def on_progress(filename, current, total):
            app.after(0, lambda: app.set_status(f"Скачивание: {filename} ({current}/{total})"))

        if not download_models(MODEL_DIR, model_type, progress_callback=on_progress):
            # Fallback to v3_rnnt if the selected model can't be downloaded
            if model_type != "v3_rnnt" and check_models_exist(MODEL_DIR, "v3_rnnt"):
                logger.warning("Failed to download %s, falling back to v3_rnnt", model_type)
                model_type = "v3_rnnt"
                settings["model"] = model_type
                app.save()
            else:
                from tkinter import messagebox
                messagebox.showerror(
                    "Ошибка скачивания",
                    f"Не удалось скачать модели {model_type}.\n\n"
                    "Возможные причины:\n"
                    "- Нет подключения к интернету\n"
                    "- Проблемы с доступом к huggingface.co\n\n"
                    "Проверьте сеть и перезапустите приложение.",
                    parent=app,
                )
                sys.exit(1)

    # --- ONNX recognizer ---
    app.set_status("Загрузка модели...")
    app.update()

    recognizer = None
    try:
        from core.onnx_recognizer import OnnxGigaAMTranscriber
        recognizer = OnnxGigaAMTranscriber(
            model_dir=MODEL_DIR, model_type=model_type, prefer_cuda=True
        )
        provider = recognizer.providers[0] if recognizer.providers else "CPU"
        logger.info("ONNX recognizer loaded on %s", provider)
        app.set_status(f"Модель загружена ({provider})")
    except Exception as e:
        logger.error("Failed to load recognizer: %s", e)
        app.set_status(f"Ошибка модели: {e}")

    # --- Recording overlay ---
    overlay = RecordingOverlay(app)

    # --- Audio ---
    recorder = AudioRecorder(samplerate=16000)

    mic_ok = True
    try:
        import sounddevice as sd
        sd.query_devices(kind="input")
    except sd.PortAudioError:
        mic_ok = False
        from tkinter import messagebox
        messagebox.showwarning(
            "Микрофон",
            "Микрофон не найден!\n\n"
            "Подключите микрофон и перезапустите приложение.\n"
            "Без микрофона запись голоса работать не будет.",
            parent=app,
        )
        logger.warning("No input audio device found")
    except Exception as e:
        logger.warning("Could not check audio devices: %s", e)

    # --- Groq ---
    groq_keys = settings.get("groq", {}).get("keys", [])
    key_pool = GroqKeyPool(groq_keys)
    groq_client = GroqClient(key_pool)

    # --- Tabs ---
    from core import text_utils

    test_tab = TestTab(app.notebook, app, recorder, recognizer, groq_client, text_utils, MODEL_DIR)
    app.notebook.add(test_tab, text="  Тест  ")
    app.test_tab = test_tab
    test_tab.overlay = overlay

    history_tab = HistoryTab(app.notebook, app)
    app.notebook.add(history_tab, text="  История  ")
    app.history_tab = history_tab
    test_tab.history_tab = history_tab

    # --- Multi-hotkey manager ---
    hotkey_mgr = MultiHotkeyManager()
    saved_hotkeys = settings.get("hotkeys", {})

    def on_start():
        recorder.start()
        app.after(0, lambda: app.set_status("Запись..."))
        if settings.get("sound_feedback", True):
            play_start_sound()
        if settings.get("show_recording_overlay", True):
            app.after(0, overlay.show)

    def on_stop(pipeline: str):
        result = recorder.stop()
        if settings.get("sound_feedback", True):
            play_stop_sound()
        app.after(0, overlay.hide)
        if result is None:
            app.after(0, lambda: app.set_status("Нет аудиоданных"))
            return
        audio, sr = result
        names = {
            "raw": "Распознавание...",
            "punctuation": "Распознавание + пунктуация...",
            "translate": "Распознавание + перевод...",
        }
        app.after(0, lambda: app.set_status(names.get(pipeline, "Обработка...")))
        test_tab.do_hotkey_recognize(audio, sr, pipeline)

    for slot_name, default_keys in DEFAULT_HOTKEYS.items():
        keys = saved_hotkeys.get(slot_name, default_keys)
        hotkey_mgr.register(
            name=slot_name,
            pipeline=slot_name,
            keys=keys,
            on_activate=on_start,
            on_deactivate=on_stop,
        )

    hotkeys_tab = HotkeysTab(app.notebook, app, hotkey_mgr)
    app.notebook.add(hotkeys_tab, text="  Клавиши  ")
    app.hotkeys_tab = hotkeys_tab

    groq_tab = GroqTab(app.notebook, app, key_pool)
    app.notebook.add(groq_tab, text="  Groq  ")
    app.groq_tab = groq_tab

    # --- Start hotkeys ---
    try:
        hotkey_mgr.start()
    except Exception as e:
        from tkinter import messagebox
        logger.error("Failed to start hotkeys: %s", e)
        messagebox.showwarning(
            "Горячие клавиши",
            "Не удалось запустить горячие клавиши.\n\n"
            "Возможные причины:\n"
            "- Нужны права администратора\n"
            "- Другая программа блокирует доступ\n\n"
            "Попробуйте запустить от имени администратора.\n"
            "Запись через кнопку в интерфейсе всё равно будет работать.",
            parent=app,
        )

    keys_display = " | ".join(
        f"{n}: {'+'.join(hotkey_mgr.get_keys(n))}"
        for n in DEFAULT_HOTKEYS
    )

    if not groq_keys:
        app.set_status(f"Готово | Groq не настроен (вкладка Groq) | {keys_display}")
    else:
        app.set_status(f"Готово | {keys_display}")

    # --- First launch guide ---
    if app.is_first_run:
        app.save()  # create settings.json with defaults
        from tkinter import messagebox
        messagebox.showinfo(
            "Добро пожаловать!",
            "Как пользоваться:\n\n"
            "1. Удерживайте горячие клавиши для записи голоса,\n"
            "   отпустите для распознавания\n\n"
            f"   {keys_display}\n\n"
            "2. Текст автоматически вставится в активное окно\n\n"
            "3. Для пунктуации и перевода (Groq) добавьте\n"
            "   бесплатный API ключ во вкладке Groq\n"
            "   (console.groq.com)\n\n"
            "4. Кнопка X сворачивает в трей, а не закрывает",
            parent=app,
        )

    # --- Tray & close behavior ---
    def do_full_quit():
        hotkey_mgr.stop()
        app.destroy()

    app.setup_tray(on_quit_callback=do_full_quit)

    # X button -> minimize to tray (tray keeps running)
    app.protocol("WM_DELETE_WINDOW", app.minimize_to_tray)

    # Start minimized if --minimized flag
    if args.minimized:
        app.after(100, app.minimize_to_tray)

    try:
        app.mainloop()
    except KeyboardInterrupt:
        hotkey_mgr.stop()


if __name__ == "__main__":
    main()
