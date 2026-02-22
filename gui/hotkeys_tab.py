import logging
import os
import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import List, Optional

import keyboard as kb_lib

logger = logging.getLogger(__name__)

# Windows autostart registry key
_AUTOSTART_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_AUTOSTART_NAME = "VoiceRecognition"

SLOT_DEFS = [
    ("raw", "1. Распознавание + вставка", "Просто распознать и сразу вставить текст"),
    ("punctuation", "2. Распознавание + пунктуация", "Распознать, добавить пунктуацию (Groq), вставить"),
    ("translate", "3. Распознавание + перевод EN", "Распознать, пунктуация + перевод на English (Groq), вставить"),
]

DEFAULT_KEYS = {
    "raw": ["ctrl", "alt"],
    "punctuation": ["ctrl", "shift"],
    "translate": ["ctrl", "shift", "alt"],
}


class HotkeysTab(ttk.Frame):
    """Hotkey configuration tab with 3 slots."""

    def __init__(self, parent, app, hotkey_manager):
        super().__init__(parent)
        self.app = app
        self.hotkey_manager = hotkey_manager

        self._capturing_slot: Optional[str] = None
        self._captured_keys: set = set()
        self._slot_widgets = {}  # slot_name -> {display, capture_btn, capture_label}

        self._build_ui()

    def _build_ui(self):
        # --- Hotkey slots ---
        for slot_name, title, description in SLOT_DEFS:
            frame = ttk.LabelFrame(self, text=title)
            frame.pack(fill=tk.X, padx=10, pady=(8, 2))

            # Description
            ttk.Label(frame, text=description, foreground="#888888").pack(
                anchor=tk.W, padx=10, pady=(4, 0)
            )

            row = ttk.Frame(frame)
            row.pack(fill=tk.X, padx=10, pady=5)

            display = ttk.Label(
                row,
                text=self._format_hotkey(self.hotkey_manager.get_keys(slot_name)),
                font=("Consolas", 11, "bold"),
            )
            display.pack(side=tk.LEFT, padx=(0, 15))

            capture_btn = ttk.Button(
                row,
                text="Назначить...",
                command=lambda sn=slot_name: self._start_capture(sn),
            )
            capture_btn.pack(side=tk.LEFT, padx=(0, 10))

            capture_label = ttk.Label(row, text="")
            capture_label.pack(side=tk.LEFT)

            self._slot_widgets[slot_name] = {
                "display": display,
                "capture_btn": capture_btn,
                "capture_label": capture_label,
            }

        # --- Injection mode ---
        inj_frame = ttk.LabelFrame(self, text="Режим вставки текста")
        inj_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        self.injection_var = tk.StringVar(
            value=self.app.settings.get("injection_mode", "clipboard")
        )
        ttk.Radiobutton(
            inj_frame,
            text="Буфер обмена + Ctrl+V (безопасный)",
            variable=self.injection_var,
            value="clipboard",
        ).pack(anchor=tk.W, padx=10, pady=(5, 2))
        ttk.Radiobutton(
            inj_frame,
            text="keyboard.write (быстрый, рискованный для мессенджеров)",
            variable=self.injection_var,
            value="keyboard_write",
        ).pack(anchor=tk.W, padx=10, pady=(2, 5))

        delay_frame = ttk.Frame(inj_frame)
        delay_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Label(delay_frame, text="Задержка перед вставкой (сек):").pack(side=tk.LEFT)
        self.delay_var = tk.DoubleVar(
            value=self.app.settings.get("injection_delay", 0.3)
        )
        ttk.Spinbox(
            delay_frame, from_=0.0, to=2.0, increment=0.1,
            textvariable=self.delay_var, width=5,
        ).pack(side=tk.LEFT, padx=10)

        # --- Replacements ---
        repl_frame = ttk.LabelFrame(self, text="Пост-обработка")
        repl_frame.pack(fill=tk.X, padx=10, pady=5)

        self.repl_var = tk.BooleanVar(
            value=self.app.settings.get("enable_replacements", True)
        )
        ttk.Checkbutton(
            repl_frame,
            text="Текстовые замены (запятая -> , точка -> . и т.д.)",
            variable=self.repl_var,
        ).pack(anchor=tk.W, padx=10, pady=5)

        # --- Feedback ---
        feedback_frame = ttk.LabelFrame(self, text="Обратная связь при записи")
        feedback_frame.pack(fill=tk.X, padx=10, pady=5)

        self.sound_var = tk.BooleanVar(
            value=self.app.settings.get("sound_feedback", True)
        )
        ttk.Checkbutton(
            feedback_frame, text="Звуковой сигнал (начало/конец записи)",
            variable=self.sound_var, command=self._on_feedback_toggle,
        ).pack(anchor=tk.W, padx=10, pady=(5, 2))

        self.overlay_var = tk.BooleanVar(
            value=self.app.settings.get("show_recording_overlay", True)
        )
        ttk.Checkbutton(
            feedback_frame, text="Индикатор записи на экране (красная точка)",
            variable=self.overlay_var, command=self._on_feedback_toggle,
        ).pack(anchor=tk.W, padx=10, pady=(2, 5))

        # --- Autostart ---
        auto_frame = ttk.LabelFrame(self, text="Автозапуск")
        auto_frame.pack(fill=tk.X, padx=10, pady=5)

        self.autostart_var = tk.BooleanVar(value=self._is_autostart_enabled())
        ttk.Checkbutton(
            auto_frame,
            text="Запускать при входе в Windows (в трей)",
            variable=self.autostart_var,
            command=self._toggle_autostart,
        ).pack(anchor=tk.W, padx=10, pady=5)

        # --- Save / Reset ---
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=8)

        ttk.Button(btn_frame, text="Сохранить", command=self._save).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Button(btn_frame, text="Сбросить всё", command=self._reset).pack(side=tk.LEFT)

    def _format_hotkey(self, keys: List[str]) -> str:
        if not keys:
            return "(не назначено)"
        return " + ".join(k.capitalize() for k in keys)

    # ---- Capture ----

    def _start_capture(self, slot_name: str):
        if self._capturing_slot:
            return
        self._capturing_slot = slot_name
        self._captured_keys.clear()

        w = self._slot_widgets[slot_name]
        w["capture_btn"].configure(text="Нажмите...")
        w["capture_label"].configure(text="(отпустите для сохранения)")

        self.hotkey_manager.stop()
        kb_lib.hook(self._on_capture_event)

    def _on_capture_event(self, event):
        if not self._capturing_slot:
            return
        name = (event.name or "").lower()
        if not name:
            return
        if event.event_type == kb_lib.KEY_DOWN:
            self._captured_keys.add(name)
            self.after(0, self._update_capture_display)
        elif event.event_type == kb_lib.KEY_UP and self._captured_keys:
            self.after(0, self._finish_capture)

    def _update_capture_display(self):
        if self._capturing_slot and self._captured_keys:
            w = self._slot_widgets[self._capturing_slot]
            w["capture_btn"].configure(
                text=self._format_hotkey(sorted(self._captured_keys))
            )

    def _finish_capture(self):
        slot_name = self._capturing_slot
        if not slot_name:
            return
        self._capturing_slot = None
        kb_lib.unhook_all()

        keys = sorted(self._captured_keys)
        w = self._slot_widgets[slot_name]

        if keys:
            self.hotkey_manager.update_keys(slot_name, keys)
            w["display"].configure(text=self._format_hotkey(keys))
            self.app.settings.setdefault("hotkeys", {})[slot_name] = keys
            self.app.save()

        w["capture_btn"].configure(text="Назначить...")
        w["capture_label"].configure(text="")
        self.hotkey_manager.start()

    # ---- Feedback ----

    def _on_feedback_toggle(self):
        self.app.settings["sound_feedback"] = self.sound_var.get()
        self.app.settings["show_recording_overlay"] = self.overlay_var.get()
        self.app.save()

    # ---- Save / Reset ----

    def _save(self):
        self.app.settings["injection_mode"] = self.injection_var.get()
        self.app.settings["injection_delay"] = self.delay_var.get()
        self.app.settings["enable_replacements"] = self.repl_var.get()
        self.app.settings["sound_feedback"] = self.sound_var.get()
        self.app.settings["show_recording_overlay"] = self.overlay_var.get()
        self.app.save()
        self.app.set_status("Настройки сохранены")

    def _reset(self):
        for slot_name, default_keys in DEFAULT_KEYS.items():
            self.hotkey_manager.update_keys(slot_name, default_keys)
            w = self._slot_widgets[slot_name]
            w["display"].configure(text=self._format_hotkey(default_keys))
            self.app.settings.setdefault("hotkeys", {})[slot_name] = default_keys

        self.injection_var.set("clipboard")
        self.delay_var.set(0.3)
        self.repl_var.set(True)
        self.sound_var.set(True)
        self.overlay_var.set(True)
        self._save()
        self.hotkey_manager.start()
        self.app.set_status("Все настройки сброшены")

    # ---- Autostart ----

    def _get_autostart_command(self) -> str:
        """Build the command string for autostart registry entry."""
        # Use pythonw.exe (no console) from the venv
        root = Path(__file__).resolve().parent.parent
        venv_pythonw = root / ".venv" / "Scripts" / "pythonw.exe"
        main_py = root / "main.py"

        if venv_pythonw.exists():
            python = str(venv_pythonw)
        else:
            python = sys.executable.replace("python.exe", "pythonw.exe")

        return f'"{python}" "{main_py}" --minimized'

    def _is_autostart_enabled(self) -> bool:
        if sys.platform != "win32":
            return False
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, _AUTOSTART_KEY, 0, winreg.KEY_READ
            )
            try:
                winreg.QueryValueEx(key, _AUTOSTART_NAME)
                return True
            except FileNotFoundError:
                return False
            finally:
                winreg.CloseKey(key)
        except Exception:
            return False

    def _toggle_autostart(self):
        if sys.platform != "win32":
            self.app.set_status("Автозапуск поддерживается только на Windows")
            return

        import winreg

        if self.autostart_var.get():
            # Enable
            try:
                key = winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER, _AUTOSTART_KEY, 0, winreg.KEY_SET_VALUE
                )
                cmd = self._get_autostart_command()
                winreg.SetValueEx(key, _AUTOSTART_NAME, 0, winreg.REG_SZ, cmd)
                winreg.CloseKey(key)
                logger.info("Autostart enabled: %s", cmd)
                self.app.set_status("Автозапуск включён")
            except Exception as e:
                logger.error("Failed to enable autostart: %s", e)
                self.autostart_var.set(False)
                self.app.set_status(f"Ошибка автозапуска: {e}")
        else:
            # Disable
            try:
                key = winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER, _AUTOSTART_KEY, 0, winreg.KEY_SET_VALUE
                )
                try:
                    winreg.DeleteValue(key, _AUTOSTART_NAME)
                except FileNotFoundError:
                    pass
                winreg.CloseKey(key)
                logger.info("Autostart disabled")
                self.app.set_status("Автозапуск выключен")
            except Exception as e:
                logger.error("Failed to disable autostart: %s", e)
                self.app.set_status(f"Ошибка: {e}")
