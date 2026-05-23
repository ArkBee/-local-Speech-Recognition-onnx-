import logging
import sys
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import keyboard

logger = logging.getLogger(__name__)

_WIN_KEY_NAMES = {"windows", "left windows", "right windows", "win", "left win", "right win"}


def _suppress_start_menu():
    """Inject a phantom F15 key event so Windows doesn't open the Start menu
    when the Win key is released as part of a hotkey combo.

    Windows opens Start menu when Win is pressed-and-released without any other
    key in between. Sending a harmless key while Win is still held breaks that
    pattern. F15 is a valid VK but virtually never bound to anything.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        VK_F15 = 0x7E
        KEYEVENTF_KEYUP = 0x0002
        user32 = ctypes.windll.user32
        user32.keybd_event(VK_F15, 0, 0, 0)
        user32.keybd_event(VK_F15, 0, KEYEVENTF_KEYUP, 0)
    except Exception:
        logger.exception("Failed to inject phantom key for Win suppression")

# Pipeline modes triggered by each hotkey slot
PIPELINE_RAW = "raw"                # recognize -> paste
PIPELINE_PUNCTUATION = "punctuation" # recognize -> groq punctuation -> paste
PIPELINE_TRANSLATE = "translate"     # recognize -> groq punctuation + translate EN -> paste


@dataclass
class HotkeySlot:
    name: str
    pipeline: str
    required: Set[str]
    on_activate: Callable[[], None]
    on_deactivate: Callable[[str], None]  # receives pipeline mode


class MultiHotkeyManager:
    """Manages multiple hotkey slots simultaneously.

    Only one slot can be active at a time.  When multiple slots match
    the currently pressed keys, the most specific one (most keys) wins.
    """

    def __init__(self):
        self.slots: Dict[str, HotkeySlot] = {}
        self._pressed: Set[str] = set()
        self._active_slot: Optional[HotkeySlot] = None
        self._hook_installed = False
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        pipeline: str,
        keys: List[str],
        on_activate: Callable[[], None],
        on_deactivate: Callable[[str], None],
    ):
        normalized = self._normalize(keys)
        if not normalized:
            raise ValueError(f"Hotkey '{name}' must have at least one key")
        slot = HotkeySlot(
            name=name,
            pipeline=pipeline,
            required=normalized,
            on_activate=on_activate,
            on_deactivate=on_deactivate,
        )
        self.slots[name] = slot
        logger.info(
            "Registered hotkey '%s' (%s): %s",
            name, pipeline, "+".join(sorted(normalized)),
        )

    def update_keys(self, name: str, keys: List[str]):
        if name not in self.slots:
            return
        normalized = self._normalize(keys)
        if not normalized:
            raise ValueError("Hotkey must have at least one key")
        with self._lock:
            was_active = self._active_slot and self._active_slot.name == name
            self.slots[name].required = normalized
            if was_active:
                self._active_slot = None
        if was_active:
            self.slots[name].on_deactivate(self.slots[name].pipeline)
        logger.info(
            "Updated hotkey '%s': %s", name, "+".join(sorted(normalized)),
        )

    def get_keys(self, name: str) -> List[str]:
        if name in self.slots:
            return sorted(self.slots[name].required)
        return []

    def _normalize(self, keys: List[str]) -> Set[str]:
        return {k.strip().lower() for k in keys if k and k.strip()}

    def _sorted_slots(self) -> List[HotkeySlot]:
        """Slots sorted by specificity (most keys first)."""
        return sorted(self.slots.values(), key=lambda s: len(s.required), reverse=True)

    def _on_key_event(self, event: keyboard.KeyboardEvent):
        raw = (event.name or "").lower()
        if not raw:
            return

        is_down = event.event_type == keyboard.KEY_DOWN
        deactivated_slot: Optional[HotkeySlot] = None
        activated_slot: Optional[HotkeySlot] = None

        with self._lock:
            if is_down:
                self._pressed.add(raw)
            else:
                self._pressed.discard(raw)

            # Deactivation: any key release that breaks the active combo.
            # No auto-upgrade, no fall-through to a smaller slot — releases
            # NEVER start a new recording.
            if self._active_slot:
                if not self._active_slot.required.issubset(self._pressed):
                    deactivated_slot = self._active_slot
                    self._active_slot = None
                # else: still active, slot stays as-is (no upgrade).
            elif is_down:
                # Only KEY_DOWN events can activate a slot. Pick the most
                # specific combo that fully matches the currently held keys.
                for slot in self._sorted_slots():
                    if slot.required.issubset(self._pressed):
                        self._active_slot = slot
                        activated_slot = slot
                        break

        # Callbacks fire OUTSIDE the lock so recorder.stop()/start() can't
        # block subsequent keyboard events.
        if deactivated_slot is not None:
            try:
                deactivated_slot.on_deactivate(deactivated_slot.pipeline)
            except Exception:
                logger.exception(
                    "Error in on_deactivate for '%s'", deactivated_slot.name
                )

        if activated_slot is not None:
            # If the combo includes Win, inject a phantom key so the Start
            # menu doesn't pop when the user releases Win last.
            if activated_slot.required & _WIN_KEY_NAMES:
                _suppress_start_menu()
            try:
                activated_slot.on_activate()
            except Exception:
                logger.exception(
                    "Error in on_activate for '%s'", activated_slot.name
                )

    def start(self):
        if self._hook_installed:
            return
        keyboard.hook(self._on_key_event)
        self._hook_installed = True
        names = [f"{s.name}({'+'.join(sorted(s.required))})" for s in self.slots.values()]
        logger.info("Hotkey listener started: %s", ", ".join(names))

    def stop(self):
        if not self._hook_installed:
            return
        keyboard.unhook_all()
        self._hook_installed = False
        self._pressed.clear()
        self._active_slot = None
        logger.info("Hotkey listener stopped")
