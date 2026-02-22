import json
import logging
import sys
import threading
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    import pystray
    from PIL import Image, ImageDraw
    _TRAY_AVAILABLE = True
except ImportError:
    _TRAY_AVAILABLE = False
    logger.warning("pystray/Pillow not installed — tray support disabled")

SETTINGS_FILE = Path(__file__).resolve().parent.parent / "settings.json"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "model": "v3_rnnt",
    "hotkeys": {
        "raw": ["ctrl", "alt"],
        "punctuation": ["ctrl", "shift"],
        "translate": ["ctrl", "shift", "alt"],
    },
    "injection_mode": "clipboard",
    "injection_delay": 0.3,
    "enable_replacements": True,
    "fuzzy_numeric": True,
    "replace_punctuation": True,
    "fuzzy_cutoff": 0.85,
    "min_fuzzy_length": 4,
    "sound_feedback": True,
    "show_recording_overlay": True,
    "custom_replacements": {},
    "groq": {
        "keys": [],
        "stt_model": "whisper-large-v3",
        "text_model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "punctuation_prompt": "Добавь пунктуацию к следующему русскому тексту. Верни только исправленный текст без пояснений.",
        "translation_prompt": "Переведи следующий текст на {language}. Верни только перевод без пояснений.",
        "target_language": "English",
    },
}


def load_settings() -> tuple:
    """Load settings, returns (settings_dict, is_first_run)."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # Merge with defaults for any missing keys
            merged = {**DEFAULT_SETTINGS, **saved}
            merged["groq"] = {**DEFAULT_SETTINGS["groq"], **saved.get("groq", {})}
            merged["hotkeys"] = {**DEFAULT_SETTINGS["hotkeys"], **saved.get("hotkeys", {})}
            return merged, False
        except Exception as e:
            logger.error("Failed to load settings: %s", e)
    return dict(DEFAULT_SETTINGS), True


def save_settings(settings: Dict[str, Any]):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error("Failed to save settings: %s", e)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Voice Recognition - GigaAM v3 + Groq")
        self.geometry("850x650")
        self.minsize(700, 550)

        self.settings, self.is_first_run = load_settings()

        self._apply_dark_theme()

        # Notebook (tabs)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 0))

        # Tabs will be added by main.py after initializing core components
        self.test_tab = None
        self.hotkeys_tab = None
        self.groq_tab = None

        # Status bar
        self.status_var = tk.StringVar(value="Готово")
        status_bar = ttk.Label(
            self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=6, pady=4)

    def _apply_dark_theme(self):
        self.configure(bg="#1e1e1e")

        style = ttk.Style(self)
        style.theme_use("clam")

        bg = "#1e1e1e"
        fg = "#d4d4d4"
        accent = "#264f78"
        field_bg = "#2d2d2d"
        select_bg = "#37373d"
        btn_bg = "#333333"

        style.configure(".", background=bg, foreground=fg, fieldbackground=field_bg)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TFrame", background=bg)
        style.configure("TButton", background=btn_bg, foreground=fg, padding=6)
        style.map(
            "TButton",
            background=[("active", accent), ("pressed", accent)],
            foreground=[("active", "#ffffff")],
        )
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", background=btn_bg, foreground=fg, padding=[12, 4])
        style.map(
            "TNotebook.Tab",
            background=[("selected", accent)],
            foreground=[("selected", "#ffffff")],
        )
        style.configure("TCheckbutton", background=bg, foreground=fg)
        style.map("TCheckbutton", background=[("active", bg)])
        style.configure("TRadiobutton", background=bg, foreground=fg)
        style.map("TRadiobutton", background=[("active", bg)])
        style.configure("TLabelframe", background=bg, foreground=fg)
        style.configure("TLabelframe.Label", background=bg, foreground=fg)
        style.configure("TCombobox", fieldbackground=field_bg, foreground=fg)
        style.configure("TEntry", fieldbackground=field_bg, foreground=fg)
        style.configure("TScale", background=bg, troughcolor=field_bg)
        style.configure("Horizontal.TProgressbar", background=accent, troughcolor=field_bg)
        style.configure("Level.Horizontal.TProgressbar", background="#44cc44", troughcolor=field_bg)

        # Treeview (history tab)
        style.configure(
            "Treeview",
            background=field_bg, foreground=fg, fieldbackground=field_bg,
            rowheight=25,
        )
        style.configure("Treeview.Heading", background=btn_bg, foreground=fg)
        style.map(
            "Treeview",
            background=[("selected", accent)],
            foreground=[("selected", "#ffffff")],
        )

        # Recording indicator styles
        style.configure("Recording.TLabel", background=bg, foreground="#ff4444", font=("", 10, "bold"))
        style.configure("Ready.TLabel", background=bg, foreground="#44ff44", font=("", 10, "bold"))
        style.configure("Status.TLabel", background=bg, foreground=fg)

    def set_status(self, text: str):
        self.status_var.set(text)

    def save(self):
        save_settings(self.settings)

    # ----------------------------------------------------------------
    # System tray
    # ----------------------------------------------------------------

    def setup_tray(self, on_quit_callback=None):
        """Initialize system tray icon. Call after mainloop is running."""
        self._on_quit_callback = on_quit_callback
        if not _TRAY_AVAILABLE:
            return
        self._tray_icon = None
        self._create_tray_icon()

    def _make_tray_image(self) -> "Image.Image":
        """Generate a simple microphone icon for the tray."""
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        # Mic body
        d.rounded_rectangle([20, 8, 44, 38], radius=10, fill="#4fc3f7")
        # Mic arc
        d.arc([14, 20, 50, 50], start=0, end=180, fill="#4fc3f7", width=3)
        # Stand
        d.line([32, 50, 32, 58], fill="#4fc3f7", width=3)
        d.line([22, 58, 42, 58], fill="#4fc3f7", width=3)
        return img

    def _create_tray_icon(self):
        if not _TRAY_AVAILABLE:
            return
        image = self._make_tray_image()
        menu = pystray.Menu(
            pystray.MenuItem("Показать", self._tray_show, default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Выход", self._tray_quit),
        )
        self._tray_icon = pystray.Icon(
            "VoiceRecognition", image, "Voice Recognition", menu
        )
        # Run tray icon in a daemon thread
        tray_thread = threading.Thread(target=self._tray_icon.run, daemon=True)
        tray_thread.start()

    def minimize_to_tray(self):
        """Hide window and remove from taskbar completely."""
        if not self.settings.get("_tray_hint_shown", False):
            from tkinter import messagebox
            messagebox.showinfo(
                "Свернуто в трей",
                "Приложение свернулось в системный трей (область уведомлений).\n\n"
                "Горячие клавиши продолжают работать!\n\n"
                "Чтобы открыть окно: двойной клик по иконке в трее.\n"
                "Чтобы закрыть: правый клик по иконке > Выход.\n\n"
                "Это сообщение больше не появится.",
                parent=self,
            )
            self.settings["_tray_hint_shown"] = True
            self.save()
        self.withdraw()
        # On Windows: remove WS_EX_APPWINDOW so it disappears from taskbar
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            GWL_EXSTYLE = -20
            WS_EX_APPWINDOW = 0x00040000
            WS_EX_TOOLWINDOW = 0x00000080
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style = (style & ~WS_EX_APPWINDOW) | WS_EX_TOOLWINDOW
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        except Exception:
            pass

    def _tray_show(self, icon=None, item=None):
        """Restore window from tray."""
        self.after(0, self._do_show)

    def _do_show(self):
        # Restore WS_EX_APPWINDOW so it shows on taskbar again
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            GWL_EXSTYLE = -20
            WS_EX_APPWINDOW = 0x00040000
            WS_EX_TOOLWINDOW = 0x00000080
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style = (style | WS_EX_APPWINDOW) & ~WS_EX_TOOLWINDOW
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        except Exception:
            pass
        self.deiconify()
        self.lift()
        self.focus_force()

    def _tray_quit(self, icon=None, item=None):
        """Fully quit from tray menu."""
        if self._tray_icon:
            self._tray_icon.stop()
        if self._on_quit_callback:
            self.after(0, self._on_quit_callback)
        else:
            self.after(0, self.destroy)
