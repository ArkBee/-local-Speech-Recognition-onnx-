import logging
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)


class HistoryTab(ttk.Frame):
    """Tab showing recognition history with copy support."""

    MAX_ENTRIES = 50

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._entries: List[Dict] = []
        self._build_ui()

    def _build_ui(self):
        # --- Toolbar ---
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, padx=10, pady=(10, 5))

        ttk.Button(toolbar, text="Копировать", command=self._copy_selected).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Button(toolbar, text="Очистить всё", command=self._clear_all).pack(
            side=tk.LEFT
        )

        self.count_label = ttk.Label(toolbar, text="0 записей")
        self.count_label.pack(side=tk.RIGHT)

        # --- Treeview ---
        list_frame = ttk.Frame(self)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.tree = ttk.Treeview(
            list_frame,
            columns=("time", "mode", "text"),
            show="headings",
            selectmode="browse",
        )
        self.tree.heading("time", text="Время")
        self.tree.heading("mode", text="Режим")
        self.tree.heading("text", text="Текст")
        self.tree.column("time", width=70, stretch=False)
        self.tree.column("mode", width=120, stretch=False)
        self.tree.column("text", width=500)

        scrollbar = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Preview ---
        preview_frame = ttk.LabelFrame(self, text="Полный текст")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self.preview_text = tk.Text(
            preview_frame, height=4, wrap=tk.WORD,
            bg="#2d2d2d", fg="#d4d4d4", insertbackground="#d4d4d4",
            selectbackground="#264f78", font=("Consolas", 11),
        )
        self.preview_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", lambda e: self._copy_selected())

    def add_entry(self, text: str, pipeline: str, duration: float):
        """Add a new recognition result to history."""
        now = datetime.now().strftime("%H:%M:%S")
        mode_names = {
            "raw": "Текст",
            "punctuation": "Пунктуация",
            "translate": "Перевод",
        }
        mode = mode_names.get(pipeline, pipeline)

        entry = {"time": now, "mode": mode, "text": text, "duration": duration}
        self._entries.insert(0, entry)

        if len(self._entries) > self.MAX_ENTRIES:
            self._entries = self._entries[:self.MAX_ENTRIES]

        self.tree.insert(
            "", 0,
            values=(now, f"{mode} ({duration:.1f}c)", text[:150].replace("\n", " ")),
        )

        children = self.tree.get_children()
        while len(children) > self.MAX_ENTRIES:
            self.tree.delete(children[-1])
            children = self.tree.get_children()

        self.count_label.configure(text=f"{len(self._entries)} записей")

    def _on_select(self, event=None):
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        if idx < len(self._entries):
            self.preview_text.delete("1.0", tk.END)
            self.preview_text.insert("1.0", self._entries[idx]["text"])

    def _copy_selected(self):
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        if idx < len(self._entries):
            text = self._entries[idx]["text"]
            self.clipboard_clear()
            self.clipboard_append(text)
            self.app.set_status("Скопировано из истории")

    def _clear_all(self):
        self._entries.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.preview_text.delete("1.0", tk.END)
        self.count_label.configure(text="0 записей")
        self.app.set_status("История очищена")
