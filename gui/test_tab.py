import logging
import threading
import time
import tkinter as tk
from tkinter import ttk

logger = logging.getLogger(__name__)


class TestTab(ttk.Frame):
    """Test panel: record voice -> recognize -> display text + timing."""

    def __init__(self, parent, app, recorder, recognizer, groq_client, text_utils, model_dir=None):
        super().__init__(parent)
        self.app = app
        self.recorder = recorder
        self.recognizer = recognizer
        self.groq_client = groq_client
        self.text_utils = text_utils
        self.model_dir = model_dir
        self._loading_model = False
        self.overlay = None
        self.history_tab = None
        self._level_job = None
        self._build_ui()

    def _build_ui(self):
        # --- Recognition mode ---
        mode_frame = ttk.LabelFrame(self, text="Режим распознавания")
        mode_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        self.mode_var = tk.StringVar(value="local")
        ttk.Radiobutton(
            mode_frame, text="Local (GigaAM ONNX)", variable=self.mode_var, value="local"
        ).pack(side=tk.LEFT, padx=10, pady=5)
        ttk.Radiobutton(
            mode_frame, text="Cloud (Groq Whisper)", variable=self.mode_var, value="groq"
        ).pack(side=tk.LEFT, padx=10, pady=5)

        # --- Local model selector ---
        model_row = ttk.Frame(mode_frame)
        model_row.pack(fill=tk.X, padx=10, pady=(0, 5))

        ttk.Label(model_row, text="Модель:").pack(side=tk.LEFT, padx=(0, 5))

        from core.model_downloader import MODEL_LABELS
        self._model_key_list = list(MODEL_LABELS.keys())
        self._model_label_list = list(MODEL_LABELS.values())
        self._label_to_key = dict(zip(self._model_label_list, self._model_key_list))

        current_model = self.app.settings.get("model", "v3_rnnt")
        current_label = MODEL_LABELS.get(current_model, self._model_label_list[0])

        self.model_var = tk.StringVar(value=current_label)
        self.model_combo = ttk.Combobox(
            model_row, textvariable=self.model_var, width=28, state="readonly",
            values=self._model_label_list,
        )
        self.model_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)

        # --- Controls ---
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=5)

        self.record_btn = ttk.Button(
            ctrl_frame, text="Записать", command=self._toggle_recording
        )
        self.record_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.status_label = ttk.Label(ctrl_frame, text="Готово", style="Ready.TLabel")
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.level_bar = ttk.Progressbar(
            ctrl_frame, style="Level.Horizontal.TProgressbar",
            maximum=100, mode="determinate",
        )
        self.level_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.time_label = ttk.Label(ctrl_frame, text="Время: —")
        self.time_label.pack(side=tk.RIGHT)

        # --- Raw text ---
        raw_frame = ttk.LabelFrame(self, text="Распознанный текст")
        raw_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.raw_text = tk.Text(
            raw_frame, height=5, wrap=tk.WORD,
            bg="#2d2d2d", fg="#d4d4d4", insertbackground="#d4d4d4",
            selectbackground="#264f78", font=("Consolas", 11),
        )
        self.raw_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Groq processing options (for manual test button) ---
        proc_frame = ttk.Frame(self)
        proc_frame.pack(fill=tk.X, padx=10, pady=2)

        self.punct_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            proc_frame, text="Пунктуация (Groq)", variable=self.punct_var
        ).pack(side=tk.LEFT, padx=(0, 15))

        self.translate_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            proc_frame, text="Перевод (Groq)", variable=self.translate_var
        ).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(proc_frame, text="Язык:").pack(side=tk.LEFT, padx=(10, 2))
        self.lang_var = tk.StringVar(
            value=self.app.settings["groq"].get("target_language", "English")
        )
        ttk.Combobox(
            proc_frame, textvariable=self.lang_var, width=14,
            values=["English", "Russian", "German", "French", "Spanish",
                    "Chinese", "Japanese", "Korean"],
        ).pack(side=tk.LEFT)

        self.groq_time_label = ttk.Label(proc_frame, text="Groq: —")
        self.groq_time_label.pack(side=tk.RIGHT)

        # --- Text processing settings ---
        repl_frame = ttk.LabelFrame(self, text="Настройки замен")
        repl_frame.pack(fill=tk.X, padx=10, pady=5)

        row_toggles = ttk.Frame(repl_frame)
        row_toggles.pack(fill=tk.X, padx=10, pady=(5, 2))

        self.enable_repl_var = tk.BooleanVar(
            value=self.app.settings.get("enable_replacements", True)
        )
        ttk.Checkbutton(
            row_toggles, text="Замены слов (цифры, символы)",
            variable=self.enable_repl_var, command=self._on_repl_toggle,
        ).pack(side=tk.LEFT, padx=(0, 15))

        self.fuzzy_var = tk.BooleanVar(
            value=self.app.settings.get("fuzzy_numeric", True)
        )
        ttk.Checkbutton(
            row_toggles, text="Нечёткий поиск чисел",
            variable=self.fuzzy_var, command=self._on_repl_toggle,
        ).pack(side=tk.LEFT, padx=(0, 15))

        self.punct_repl_var = tk.BooleanVar(
            value=self.app.settings.get("replace_punctuation", True)
        )
        ttk.Checkbutton(
            row_toggles, text="Замена пунктуации",
            variable=self.punct_repl_var, command=self._on_repl_toggle,
        ).pack(side=tk.LEFT)

        ttk.Button(
            row_toggles, text="Свои замены...",
            command=self._open_custom_replacements,
        ).pack(side=tk.RIGHT)

        row_sliders = ttk.Frame(repl_frame)
        row_sliders.pack(fill=tk.X, padx=10, pady=(2, 5))

        ttk.Label(row_sliders, text="Мин. длина слова:").pack(side=tk.LEFT, padx=(0, 5))
        self.min_len_var = tk.IntVar(
            value=self.app.settings.get("min_fuzzy_length", 4)
        )
        self.min_len_label = ttk.Label(row_sliders, text=str(self.min_len_var.get()), width=2)
        min_len_scale = ttk.Scale(
            row_sliders, from_=2, to=7, variable=self.min_len_var,
            orient=tk.HORIZONTAL, length=100,
            command=lambda v: self._on_slider_change(),
        )
        min_len_scale.pack(side=tk.LEFT, padx=(0, 2))
        self.min_len_label.pack(side=tk.LEFT, padx=(0, 20))

        ttk.Label(row_sliders, text="Точность:").pack(side=tk.LEFT, padx=(0, 5))
        self.cutoff_var = tk.DoubleVar(
            value=self.app.settings.get("fuzzy_cutoff", 0.85)
        )
        self.cutoff_label = ttk.Label(
            row_sliders, text=f"{self.cutoff_var.get():.2f}", width=4
        )
        cutoff_scale = ttk.Scale(
            row_sliders, from_=0.70, to=1.00, variable=self.cutoff_var,
            orient=tk.HORIZONTAL, length=100,
            command=lambda v: self._on_slider_change(),
        )
        cutoff_scale.pack(side=tk.LEFT, padx=(0, 2))
        self.cutoff_label.pack(side=tk.LEFT)

        # --- Processed text ---
        proc_text_frame = ttk.LabelFrame(self, text="Обработанный текст (Groq)")
        proc_text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.proc_text = tk.Text(
            proc_text_frame, height=5, wrap=tk.WORD,
            bg="#2d2d2d", fg="#d4d4d4", insertbackground="#d4d4d4",
            selectbackground="#264f78", font=("Consolas", 11),
        )
        self.proc_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Action buttons ---
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=(2, 10))

        ttk.Button(btn_frame, text="Копировать", command=self._copy_result).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Button(
            btn_frame, text="Вставить в активное окно", command=self._paste_to_window
        ).pack(side=tk.LEFT)

    # ----------------------------------------------------------------
    # Text processing settings
    # ----------------------------------------------------------------

    def _on_repl_toggle(self):
        self.app.settings["enable_replacements"] = self.enable_repl_var.get()
        self.app.settings["fuzzy_numeric"] = self.fuzzy_var.get()
        self.app.settings["replace_punctuation"] = self.punct_repl_var.get()
        self.app.save()

    def _on_slider_change(self):
        val_len = self.min_len_var.get()
        val_cutoff = self.cutoff_var.get()
        self.min_len_label.configure(text=str(val_len))
        self.cutoff_label.configure(text=f"{val_cutoff:.2f}")
        self.app.settings["min_fuzzy_length"] = val_len
        self.app.settings["fuzzy_cutoff"] = round(val_cutoff, 2)
        self.app.save()

    # ----------------------------------------------------------------
    # Model switching
    # ----------------------------------------------------------------

    def _on_model_change(self, event=None):
        label = self.model_var.get()
        model_key = self._label_to_key.get(label)
        if not model_key or model_key == self.app.settings.get("model"):
            return
        if self._loading_model:
            return

        logger.info("Switching model to: %s", model_key)
        self.app.settings["model"] = model_key
        self.app.save()
        self._loading_model = True
        self.model_combo.configure(state="disabled")
        self.record_btn.configure(state="disabled")
        self.status_label.configure(text="Смена модели...", style="Recording.TLabel")
        self.app.set_status(f"Переключение на {model_key}...")
        threading.Thread(target=self._load_model, args=(model_key,), daemon=True).start()

    def _set_model_status(self, text, style="Recording.TLabel", status=None):
        """Update both the status label and status bar."""
        self.status_label.configure(text=text, style=style)
        self.app.set_status(status or text)

    def _load_model(self, model_type: str):
        from core.model_downloader import check_models_exist, download_models

        try:
            if self.model_dir and not check_models_exist(self.model_dir, model_type):
                self.after(0, lambda: self._set_model_status(
                    f"Скачивание {model_type}..."))

                def on_progress(fn, cur, total):
                    msg = f"Скачивание: {fn} ({cur}/{total})"
                    self.after(0, lambda: self._set_model_status(msg))

                if not download_models(self.model_dir, model_type, progress_callback=on_progress):
                    self.after(0, lambda: self._set_model_status(
                        f"Ошибка скачивания!", "Recording.TLabel",
                        f"Не удалось скачать {model_type}"))
                    return

            self.after(0, lambda: self._set_model_status(
                f"Загрузка {model_type}..."))

            from core.onnx_recognizer import OnnxGigaAMTranscriber
            new_recognizer = OnnxGigaAMTranscriber(
                model_dir=self.model_dir, model_type=model_type, prefer_cuda=True
            )
            self.recognizer = new_recognizer
            provider = new_recognizer.providers[0] if new_recognizer.providers else "CPU"
            logger.info("Model %s loaded on %s", model_type, provider)
            self.after(0, lambda: self._set_model_status(
                f"{model_type} ({provider})", "Ready.TLabel",
                f"Модель {model_type} загружена ({provider})"))
        except Exception as e:
            logger.exception("Failed to load model %s", model_type)
            self.after(0, lambda: self._set_model_status(
                "Ошибка загрузки!", "Recording.TLabel",
                f"Ошибка модели: {e}"))
        finally:
            self._loading_model = False
            self.after(0, lambda: self.model_combo.configure(state="readonly"))
            self.after(0, lambda: self.record_btn.configure(state="normal"))

    # ----------------------------------------------------------------
    # Manual test (GUI button)
    # ----------------------------------------------------------------

    def _toggle_recording(self):
        if self.recorder.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        self.recorder.start()
        self.record_btn.configure(text="Стоп")
        self.status_label.configure(text="Запись...", style="Recording.TLabel")
        self.app.set_status("Идёт запись...")
        self._start_level_poll()
        if self.overlay and self.app.settings.get("show_recording_overlay", True):
            self.overlay.show()
        if self.app.settings.get("sound_feedback", True):
            from gui.recording_overlay import play_start_sound
            play_start_sound()

    def _stop_recording(self):
        result = self.recorder.stop()
        self.record_btn.configure(text="Записать")
        self.status_label.configure(text="Обработка...", style="Status.TLabel")
        self.app.set_status("Распознавание...")
        self._stop_level_poll()
        if self.overlay:
            self.overlay.hide()
        if self.app.settings.get("sound_feedback", True):
            from gui.recording_overlay import play_stop_sound
            play_stop_sound()

        if result is None:
            self.status_label.configure(text="Нет данных", style="Ready.TLabel")
            self.app.set_status("Нет аудиоданных")
            return

        audio, sr = result
        # Build pipeline from checkboxes
        pipeline = "raw"
        if self.translate_var.get():
            pipeline = "translate"
        elif self.punct_var.get():
            pipeline = "punctuation"

        threading.Thread(
            target=self._do_recognition, args=(audio, sr, pipeline, False),
            daemon=True,
        ).start()

    # ----------------------------------------------------------------
    # Hotkey-triggered recognition (called from main.py)
    # ----------------------------------------------------------------

    def do_hotkey_recognize(self, audio, sr, pipeline: str):
        """Called from hotkey manager. Runs full pipeline + injects text."""
        threading.Thread(
            target=self._do_recognition, args=(audio, sr, pipeline, True),
            daemon=True,
        ).start()

    # ----------------------------------------------------------------
    # Core recognition pipeline
    # ----------------------------------------------------------------

    def _do_recognition(self, audio, sr, pipeline: str, inject: bool):
        """
        pipeline: "raw" | "punctuation" | "translate"
        inject: if True, paste text into active window after recognition
        """
        try:
            mode = self.mode_var.get()
            t0 = time.perf_counter()

            if mode == "groq":
                raw = self.groq_client.transcribe(audio, sr)
            else:
                raw = self.recognizer.transcribe(audio, sr)

            recog_time = time.perf_counter() - t0

            if self.app.settings.get("enable_replacements", True):
                from core.text_utils import process_transcription
                custom_repl = self.app.settings.get("custom_replacements", {})
                raw = process_transcription(
                    raw,
                    fuzzy_numeric=self.app.settings.get("fuzzy_numeric", True),
                    fuzzy_cutoff=self.app.settings.get("fuzzy_cutoff", 0.85),
                    min_fuzzy_length=self.app.settings.get("min_fuzzy_length", 4),
                    replace_punctuation=self.app.settings.get("replace_punctuation", True),
                    custom_replacements=custom_repl if custom_repl else None,
                )

            self.after(0, self._show_raw_text, raw, recog_time)

            if not raw:
                self.after(0, lambda: self.app.set_status("Пустой результат"))
                return

            # Groq post-processing based on pipeline
            text = raw
            groq_time = 0.0

            if pipeline in ("punctuation", "translate"):
                self.after(0, lambda: self.app.set_status("Groq обработка..."))
                t1 = time.perf_counter()

                prompt = self.app.settings["groq"].get(
                    "punctuation_prompt",
                    "Добавь пунктуацию к следующему русскому тексту. "
                    "Верни только исправленный текст без пояснений.",
                )
                model = self.app.settings["groq"].get(
                    "text_model", "meta-llama/llama-4-scout-17b-16e-instruct"
                )
                text = self.groq_client.add_punctuation(text, prompt, model)

                if pipeline == "translate":
                    lang = self.lang_var.get() or "English"
                    trans_prompt = self.app.settings["groq"].get(
                        "translation_prompt",
                        "Переведи следующий текст на {language}. "
                        "Верни только перевод без пояснений.",
                    )
                    text = self.groq_client.translate(text, lang, trans_prompt, model)

                groq_time = time.perf_counter() - t1
                self.after(0, self._show_processed_text, text, groq_time)
            else:
                self.after(0, self._show_processed_text, "", 0)

            # Inject if triggered by hotkey
            if inject:
                final = text
                injection_mode = self.app.settings.get("injection_mode", "clipboard")
                delay = self.app.settings.get("injection_delay", 0.3)

                if injection_mode == "clipboard":
                    self.after(0, lambda: self._clipboard_paste(final, delay))
                else:
                    import keyboard as kb
                    time.sleep(delay)
                    kb.write(final)

            total = recog_time + groq_time
            pipeline_names = {"raw": "Распознано", "punctuation": "+пунктуация", "translate": "+перевод"}
            label = pipeline_names.get(pipeline, "")
            self.after(
                0,
                lambda: self.app.set_status(f"{label} за {total:.2f}с"),
            )

            # Add to history
            final_text = text if text else raw
            if final_text and self.history_tab:
                self.after(0, lambda: self.history_tab.add_entry(final_text, pipeline, total))

        except Exception as e:
            logger.exception("Recognition error")
            self.after(0, self._show_error, str(e))

    # ----------------------------------------------------------------
    # GUI helpers
    # ----------------------------------------------------------------

    def _show_raw_text(self, text: str, duration: float):
        self.raw_text.delete("1.0", tk.END)
        self.raw_text.insert("1.0", text)
        self.time_label.configure(text=f"Время: {duration:.2f}с")
        self.status_label.configure(text="Готово", style="Ready.TLabel")

    def _show_processed_text(self, text: str, duration: float):
        self.proc_text.delete("1.0", tk.END)
        if text:
            self.proc_text.insert("1.0", text)
            self.groq_time_label.configure(text=f"Groq: {duration:.2f}с")
        else:
            self.groq_time_label.configure(text="Groq: —")

    def _show_error(self, error: str):
        self.status_label.configure(text="Ошибка", style="Recording.TLabel")
        self.app.set_status(f"Ошибка: {error}")

    def _get_best_text(self) -> str:
        processed = self.proc_text.get("1.0", tk.END).strip()
        if processed:
            return processed
        return self.raw_text.get("1.0", tk.END).strip()

    def _copy_result(self):
        text = self._get_best_text()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.app.set_status("Скопировано в буфер обмена")

    def _paste_to_window(self):
        text = self._get_best_text()
        if not text:
            return
        mode = self.app.settings.get("injection_mode", "clipboard")
        delay = self.app.settings.get("injection_delay", 0.3)
        if mode == "clipboard":
            self._clipboard_paste(text, delay)
        else:
            import keyboard as kb
            time.sleep(delay)
            kb.write(text)
            self.app.set_status("Вставлено через keyboard.write")

    def _clipboard_paste(self, text: str, delay: float):
        import keyboard as kb
        self.clipboard_clear()
        self.clipboard_append(text)
        self.after(int(delay * 1000), lambda: kb.press_and_release("ctrl+v"))
        self.app.set_status("Вставлено через буфер обмена")

    # ----------------------------------------------------------------
    # Mic level meter
    # ----------------------------------------------------------------

    def _start_level_poll(self):
        self._poll_level()

    def _stop_level_poll(self):
        if self._level_job:
            self.after_cancel(self._level_job)
            self._level_job = None
        self.level_bar["value"] = 0

    def _poll_level(self):
        if self.recorder.is_recording:
            rms = self.recorder.current_rms
            level = min(100, rms / 80)
            self.level_bar["value"] = level
            self._level_job = self.after(50, self._poll_level)
        else:
            self.level_bar["value"] = 0

    # ----------------------------------------------------------------
    # Custom replacements dialog
    # ----------------------------------------------------------------

    def _open_custom_replacements(self):
        dialog = _CustomReplacementsDialog(self, self.app)
        dialog.grab_set()
        self.wait_window(dialog)


class _CustomReplacementsDialog(tk.Toplevel):
    """Dialog for editing user-defined word replacements."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.title("Свои замены")
        self.geometry("520x420")
        self.configure(bg="#1e1e1e")
        self.transient(parent)
        self.minsize(400, 300)

        ttk.Label(
            self,
            text="Формат: фраза = замена (по одной на строку)",
        ).pack(anchor=tk.W, padx=15, pady=(15, 2))

        ttk.Label(
            self,
            text="Примеры:  мой емейл = user@mail.com  |  подпись = С уважением, Иван",
            foreground="#888888",
        ).pack(anchor=tk.W, padx=15, pady=(0, 8))

        self.text = tk.Text(
            self, wrap=tk.WORD,
            bg="#2d2d2d", fg="#d4d4d4", insertbackground="#d4d4d4",
            selectbackground="#264f78", font=("Consolas", 11),
        )
        self.text.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 10))

        # Load existing replacements
        custom = self.app.settings.get("custom_replacements", {})
        for word, repl in custom.items():
            self.text.insert(tk.END, f"{word} = {repl}\n")

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=15, pady=(0, 15))

        ttk.Button(btn_frame, text="Сохранить", command=self._save).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Button(btn_frame, text="Отмена", command=self.destroy).pack(side=tk.LEFT)

        count = len(custom)
        self.count_label = ttk.Label(btn_frame, text=f"{count} замен")
        self.count_label.pack(side=tk.RIGHT)

    def _save(self):
        content = self.text.get("1.0", tk.END)
        custom = {}
        for line in content.strip().splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue
            parts = line.split("=", 1)
            word = parts[0].strip()
            repl = parts[1].strip()
            if word and repl:
                custom[word] = repl
        self.app.settings["custom_replacements"] = custom
        self.app.save()
        self.app.set_status(f"Сохранено {len(custom)} своих замен")
        self.destroy()
