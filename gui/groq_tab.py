import logging
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from typing import Optional

logger = logging.getLogger(__name__)


class GroqTab(ttk.Frame):
    """Groq API settings: keys, models, prompts."""

    def __init__(self, parent, app, groq_key_pool):
        super().__init__(parent)
        self.app = app
        self.key_pool = groq_key_pool

        self._build_ui()

    def _build_ui(self):
        # --- API Keys ---
        keys_frame = ttk.LabelFrame(self, text="API Ключи (обойма)")
        keys_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        list_frame = ttk.Frame(keys_frame)
        list_frame.pack(fill=tk.X, padx=10, pady=5)

        self.keys_listbox = tk.Listbox(
            list_frame, height=4,
            bg="#2d2d2d", fg="#d4d4d4",
            selectbackground="#264f78",
            font=("Consolas", 10),
        )
        self.keys_listbox.pack(fill=tk.X, side=tk.LEFT, expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.keys_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.keys_listbox.configure(yscrollcommand=scrollbar.set)

        btn_row = ttk.Frame(keys_frame)
        btn_row.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Button(btn_row, text="+ Добавить ключ", command=self._add_key).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Button(btn_row, text="Удалить выбранный", command=self._remove_key).pack(
            side=tk.LEFT
        )

        self._refresh_keys_list()

        self.no_keys_hint = ttk.Label(
            keys_frame,
            text="Нет ключей. Получите бесплатный ключ на console.groq.com",
            foreground="#ff8844",
        )
        if not self.key_pool.keys:
            self.no_keys_hint.pack(fill=tk.X, padx=10, pady=(0, 5))

        # --- Models ---
        model_frame = ttk.LabelFrame(self, text="Модели Groq")
        model_frame.pack(fill=tk.X, padx=10, pady=5)

        row1 = ttk.Frame(model_frame)
        row1.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(row1, text="STT модель:").pack(side=tk.LEFT, padx=(0, 5))
        self.stt_model_var = tk.StringVar(
            value=self.app.settings["groq"].get("stt_model", "whisper-large-v3")
        )
        stt_combo = ttk.Combobox(
            row1,
            textvariable=self.stt_model_var,
            values=["whisper-large-v3", "whisper-large-v3-turbo", "distil-whisper-large-v3-en"],
            width=30,
        )
        stt_combo.pack(side=tk.LEFT)

        row2 = ttk.Frame(model_frame)
        row2.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Label(row2, text="Текст модель:").pack(side=tk.LEFT, padx=(0, 5))
        self.text_model_var = tk.StringVar(
            value=self.app.settings["groq"].get(
                "text_model", "meta-llama/llama-4-scout-17b-16e-instruct"
            )
        )
        text_combo = ttk.Combobox(
            row2,
            textvariable=self.text_model_var,
            values=[
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "meta-llama/llama-4-maverick-17b-128e-instruct",
                "meta-llama/llama-3.3-70b-versatile",
                "meta-llama/llama-3.1-8b-instant",
            ],
            width=42,
        )
        text_combo.pack(side=tk.LEFT)

        # --- Punctuation prompt ---
        punct_frame = ttk.LabelFrame(self, text="Промпт пунктуации")
        punct_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.punct_text = tk.Text(
            punct_frame, height=3, wrap=tk.WORD,
            bg="#2d2d2d", fg="#d4d4d4", insertbackground="#d4d4d4",
            selectbackground="#264f78", font=("Consolas", 10),
        )
        self.punct_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.punct_text.insert(
            "1.0",
            self.app.settings["groq"].get(
                "punctuation_prompt",
                "Добавь пунктуацию к следующему русскому тексту. Верни только исправленный текст без пояснений.",
            ),
        )

        # --- Translation prompt ---
        trans_frame = ttk.LabelFrame(self, text="Промпт перевода ({language} = целевой язык)")
        trans_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.trans_text = tk.Text(
            trans_frame, height=3, wrap=tk.WORD,
            bg="#2d2d2d", fg="#d4d4d4", insertbackground="#d4d4d4",
            selectbackground="#264f78", font=("Consolas", 10),
        )
        self.trans_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.trans_text.insert(
            "1.0",
            self.app.settings["groq"].get(
                "translation_prompt",
                "Переведи следующий текст на {language}. Верни только перевод без пояснений.",
            ),
        )

        # --- Target language ---
        lang_frame = ttk.Frame(self)
        lang_frame.pack(fill=tk.X, padx=10, pady=2)

        ttk.Label(lang_frame, text="Язык перевода по умолчанию:").pack(side=tk.LEFT, padx=(0, 5))
        self.lang_var = tk.StringVar(
            value=self.app.settings["groq"].get("target_language", "English")
        )
        ttk.Combobox(
            lang_frame,
            textvariable=self.lang_var,
            values=["English", "Russian", "German", "French", "Spanish", "Chinese", "Japanese", "Korean"],
            width=14,
        ).pack(side=tk.LEFT)

        # --- Save ---
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(btn_frame, text="Сохранить всё", command=self._save_all).pack(side=tk.LEFT)

    def _refresh_keys_list(self):
        self.keys_listbox.delete(0, tk.END)
        masked = self.key_pool.get_masked_keys()
        for i, mk in enumerate(masked):
            prefix = " [активен] " if i == (self.key_pool._index - 1) % max(self.key_pool.count, 1) else "          "
            self.keys_listbox.insert(tk.END, f"  {i + 1}. {mk}{prefix}")

    def _add_key(self):
        key = simpledialog.askstring(
            "Добавить API ключ",
            "Введите Groq API ключ (gsk_...):",
            parent=self,
        )
        if key and key.strip():
            key = key.strip()
            if not key.startswith("gsk_"):
                messagebox.showwarning("Внимание", "Ключ должен начинаться с gsk_", parent=self)
                return
            self.key_pool.add_key(key)
            self._refresh_keys_list()
            self._sync_keys_to_settings()
            self.no_keys_hint.pack_forget()
            self.app.set_status(f"Добавлен ключ ({self.key_pool.count} всего)")

    def _remove_key(self):
        sel = self.keys_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx < len(self.key_pool.keys):
            self.key_pool.remove_key(self.key_pool.keys[idx])
            self._refresh_keys_list()
            self._sync_keys_to_settings()
            if not self.key_pool.keys:
                self.no_keys_hint.pack(fill=tk.X, padx=10, pady=(0, 5))
            self.app.set_status(f"Ключ удалён ({self.key_pool.count} осталось)")

    def _sync_keys_to_settings(self):
        self.app.settings["groq"]["keys"] = list(self.key_pool.keys)
        self.app.save()

    def _save_all(self):
        self._sync_keys_to_settings()
        self.app.settings["groq"]["stt_model"] = self.stt_model_var.get()
        self.app.settings["groq"]["text_model"] = self.text_model_var.get()
        self.app.settings["groq"]["punctuation_prompt"] = self.punct_text.get("1.0", tk.END).strip()
        self.app.settings["groq"]["translation_prompt"] = self.trans_text.get("1.0", tk.END).strip()
        self.app.settings["groq"]["target_language"] = self.lang_var.get()
        self.app.save()
        self.app.set_status("Настройки Groq сохранены")
