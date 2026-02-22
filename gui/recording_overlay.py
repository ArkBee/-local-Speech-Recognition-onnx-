import logging
import sys
import threading
import tkinter as tk

logger = logging.getLogger(__name__)


def play_start_sound():
    """Short high beep when recording starts."""
    if sys.platform != "win32":
        return

    def _play():
        try:
            import winsound
            winsound.Beep(800, 100)
        except Exception:
            pass

    threading.Thread(target=_play, daemon=True).start()


def play_stop_sound():
    """Lower beep when recording stops."""
    if sys.platform != "win32":
        return

    def _play():
        try:
            import winsound
            winsound.Beep(500, 150)
        except Exception:
            pass

    threading.Thread(target=_play, daemon=True).start()


class RecordingOverlay(tk.Toplevel):
    """Small always-on-top pulsing red dot shown during recording."""

    SIZE = 28
    MARGIN = 20

    def __init__(self, master):
        super().__init__(master)
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        try:
            self.attributes("-alpha", 0.85)
        except tk.TclError:
            pass

        self.configure(bg="black")
        self.canvas = tk.Canvas(
            self, width=self.SIZE, height=self.SIZE,
            bg="black", highlightthickness=0,
        )
        self.canvas.pack()

        self._dot = self.canvas.create_oval(
            3, 3, self.SIZE - 3, self.SIZE - 3,
            fill="#ff2222", outline="#ff4444", width=1,
        )

        # Position: top-right corner of screen
        screen_w = self.winfo_screenwidth()
        x = screen_w - self.SIZE - self.MARGIN
        y = self.MARGIN
        self.geometry(f"{self.SIZE}x{self.SIZE}+{x}+{y}")

        self._bright = True
        self._pulse_job = None
        self.withdraw()

    def show(self):
        self.deiconify()
        self.lift()
        self._start_pulse()

    def hide(self):
        self._stop_pulse()
        self.withdraw()

    def _start_pulse(self):
        self._bright = True
        self._pulse()

    def _stop_pulse(self):
        if self._pulse_job:
            self.after_cancel(self._pulse_job)
            self._pulse_job = None

    def _pulse(self):
        color = "#ff2222" if self._bright else "#771111"
        self.canvas.itemconfig(self._dot, fill=color)
        self._bright = not self._bright
        self._pulse_job = self.after(500, self._pulse)
