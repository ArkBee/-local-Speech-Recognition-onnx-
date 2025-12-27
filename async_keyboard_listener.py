import asyncio
import keyboard
import logging
from typing import Callable, Coroutine, Iterable, Set, List, Optional
from pathlib import Path

# Опционально поддерживаем библиотеку для мыши
try:
    import mouse  # type: ignore
    _MOUSE_AVAILABLE = True
except Exception:  # ImportError или любые другие ошибки
    mouse = None  # type: ignore
    _MOUSE_AVAILABLE = False

# Настройка логирования для этого модуля
logger = logging.getLogger(__name__)
# Чтобы видеть логи от этого модуля, если основное приложение настроит logging.basicConfig
logger.setLevel(logging.DEBUG) # Включаем DEBUG логи для этого модуля

class AsyncKeyboardListener:
    """Асинхронный слушатель клавиатуры с настраиваемой комбинацией (hotkey).

    Вместо жёстко захардкоженных Ctrl+Alt+Shift теперь можно указать любое
    множество клавиш. При одновременном зажатии ВСЕХ клавиш комбинации
    вызывается ``start_recording_callback``. Как только ЛЮБАЯ клавиша из
    комбинации отпущена – вызывается ``stop_recording_callback``.

    Поток без блокировок: библиотека ``keyboard`` вызывает обработчик в своём
    потоке, а мы пробрасываем события в asyncio через ``loop.call_soon_threadsafe``.
    """

    def __init__(
        self,
        start_recording_callback: Callable[[], Coroutine[None, None, None]],
        stop_recording_callback: Callable[[], Coroutine[None, None, None]],
        hotkey: Optional[Iterable[str]] = None,
        *,
        normalize_case: bool = True,
    ):
        """Инициализация.

        Args:
            start_recording_callback: async-функция при активации хоткея.
            stop_recording_callback: async-функция при деактивации хоткея.
            hotkey: Iterable имён клавиш (как их видит библиотека ``keyboard``)
                напр.: ("ctrl", "alt", "shift") или ("ctrl", "space").
                Если ``None`` – по умолчанию ("ctrl", "alt", "shift").
            normalize_case: Приводить ли имена клавиш к lower() (рекомендуется).
        """
        self._loop = None  # будет присвоен при старте
        self.start_recording_callback = start_recording_callback
        self.stop_recording_callback = stop_recording_callback

        if hotkey is None:
            hotkey = ("ctrl", "alt", "shift")
        self._normalize_case = normalize_case
        self._hotkey_required: Set[str] = self._normalize_hotkey(hotkey)
        if not self._hotkey_required:
            raise ValueError("Hotkey combination must contain at least one key")

        # Текущее множество нажатых клавиш (только из интересующих нас)
        self._pressed: Set[str] = set()

        # Флаг активной комбинации
        self._hotkey_currently_active = False

        self._hook_installed = False

    # ------------------------------------------------------------------
    # ВНУТРЕННИЕ УТИЛИТЫ
    # ------------------------------------------------------------------
    def _normalize_hotkey(self, hotkey: Iterable[str]) -> Set[str]:
        keys: Set[str] = set()
        for k in hotkey:
            if k is None:
                continue
            k_str = str(k).strip()
            if not k_str:
                continue
            if self._normalize_case:
                k_str = k_str.lower()
            keys.add(k_str)
        return keys

    def _hotkey_to_string(self) -> str:
        # Стабильный порядок для логов
        return "+".join(sorted(self._hotkey_required))

    def _evaluate_state_after_change(self):
        """Проверяет состояние после изменения набора нажатых клавиш/кнопок."""
        logger.debug(
            "Pressed subset changed: now=%s required=%s active=%s",
            sorted(self._pressed),
            sorted(self._hotkey_required),
            self._hotkey_currently_active,
        )
        all_pressed = self._hotkey_required.issubset(self._pressed)
        if all_pressed and not self._hotkey_currently_active:
            self._hotkey_currently_active = True
            logger.info("Hotkey activated: %s", self._hotkey_to_string())
            if self._loop is not None:
                self._loop.call_soon_threadsafe(
                    asyncio.create_task, self.start_recording_callback()
                )
        elif not all_pressed and self._hotkey_currently_active:
            self._hotkey_currently_active = False
            logger.info("Hotkey deactivated: %s", self._hotkey_to_string())
            if self._loop is not None:
                self._loop.call_soon_threadsafe(
                    asyncio.create_task, self.stop_recording_callback()
                )

    def _key_event_handler(self, event: keyboard.KeyboardEvent):
        """Низкоуровневый обработчик событий клавиатуры (поток ``keyboard``)."""
        if not self._loop:
            return
        raw_name = event.name or ""
        key_name = raw_name.lower() if self._normalize_case else raw_name
        if key_name not in self._hotkey_required:
            return
        is_key_down = event.event_type == keyboard.KEY_DOWN
        before = set(self._pressed)
        if is_key_down:
            self._pressed.add(key_name)
        else:
            self._pressed.discard(key_name)
        if before != self._pressed:
            self._evaluate_state_after_change()

    def _mouse_event_handler(self, event):  # тип из библиотеки mouse
        if not self._loop:
            return
        # Убедимся, что пользователь вообще хочет мышь
        if 'mouse_middle' not in self._hotkey_required:
            return
        try:
            etype = getattr(event, 'event_type', None)
            button = getattr(event, 'button', None)
        except Exception:
            return
        if button != 'middle' or etype not in ('up', 'down'):
            return
        before = set(self._pressed)
        if etype == 'down':
            self._pressed.add('mouse_middle')
        else:
            self._pressed.discard('mouse_middle')
        if before != self._pressed:
            self._evaluate_state_after_change()

    # ------------------------------------------------------------------
    # ПУБЛИЧНЫЕ МЕТОДЫ
    # ------------------------------------------------------------------
    def update_hotkey(self, new_hotkey: Iterable[str]):
        """Динамическая смена комбинации.

        Если комбинация была активна – вызываем stop callback (чтобы потребитель
        не остался в подвешенном состоянии) и сбрасываем состояние.
        """
        new_set = self._normalize_hotkey(new_hotkey)
        if not new_set:
            raise ValueError("New hotkey combination must contain at least one key")

        if new_set == self._hotkey_required:
            logger.debug("update_hotkey: combination is the same, skipping")
            return

        was_active = self._hotkey_currently_active
        self._hotkey_required = new_set
        self._pressed.clear()
        self._hotkey_currently_active = False

        if 'mouse_middle' in self._hotkey_required and not _MOUSE_AVAILABLE:
            raise ImportError("Requested 'mouse_middle' but package 'mouse' не установлен. Установите: pip install mouse (от админа в Windows) или уберите 'mouse_middle' из комбинации.")
        logger.info("Hotkey updated to: %s (size=%d)", self._hotkey_to_string(), len(self._hotkey_required))
        if was_active and self._loop:
            # Завершаем предыдущую активность
            self._loop.call_soon_threadsafe(
                asyncio.create_task, self.stop_recording_callback()
            )

    # ------------------------- PERSISTENCE ----------------------------
    @staticmethod
    def _parse_hotkey_file_lines(lines: List[str]) -> List[str]:
        keys: List[str] = []
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            keys.append(line)
        return keys

    @classmethod
    def load_hotkey_from_file(cls, path: str | Path) -> List[str]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Hotkey config file not found: {p}")
        lines = p.read_text(encoding='utf-8').splitlines()
        keys = cls._parse_hotkey_file_lines(lines)
        if not keys:
            raise ValueError(f"Hotkey file '{p}' does not contain any active key lines")
        return keys

    def save_hotkey_to_file(self, path: str | Path, include_template: bool = True):
        p = Path(path)
        template = ''
        if include_template:
            template = self.generate_hotkey_template(comment_only=True)
        active_part = '\n'.join(self.current_hotkey)
        content = template + ('\n' if template and not template.endswith('\n') else '') + active_part + '\n'
        p.write_text(content, encoding='utf-8')
        logger.info("Saved current hotkey (%s) to file: %s", self._hotkey_to_string(), p)

    @staticmethod
    def generate_hotkey_template(comment_only: bool = False) -> str:
        """Генерирует шаблон с наиболее часто используемыми именами клавиш.

        Если comment_only=True, возвращает только закомментированные строки (без активных ключей).
        """
        # Список часто используемых имён клавиш (адаптируемый)
        common_keys = [
            'ctrl', 'alt', 'shift', 'space', 'enter', 'esc',
            'tab', 'backspace', 'delete', 'insert', 'home', 'end',
            'page up', 'page down', 'up', 'down', 'left', 'right',
            'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
            'caps lock', 'num lock', 'print screen', 'scroll lock',
            'win', 'menu',
            # Буквы и цифры
            *[chr(c) for c in range(ord('a'), ord('z')+1)],
            *[str(d) for d in range(0, 10)],
        ]
        header = [
            '# Hotkey configuration file',
            '# Каждая непустая и не закомментированная строка = одна клавиша комбинации.',
            '# Примеры корректных имён (регистронезависимо):',
            '#   ctrl',
            '#   space',
            '#   shift',
            '#   a',
            '#   f5',
            '# Пустые или начинающиеся с # строки игнорируются.',
            '# ВНИМАНИЕ: комбинация активируется, когда одновременно зажаты ВСЕ перечисленные клавиши.',
            '# ------------- СПИСОК ЧАСТО ВСТРЕЧАЮЩИХСЯ КЛАВИШ -------------',
        ]
        key_lines = [f"# {k}" for k in common_keys]
        if comment_only:
            return '\n'.join(header + key_lines) + '\n'
        else:
            return '\n'.join(header + key_lines + [''])  # оставим пустую строку для активных

    def load_and_apply_hotkey_file(self, path: str | Path):
        keys = self.load_hotkey_from_file(path)
        self.update_hotkey(keys)
        logger.info("Applied hotkey from file '%s': %s", path, self._hotkey_to_string())

    @property
    def current_hotkey(self) -> List[str]:
        return sorted(self._hotkey_required)

    @property
    def is_active(self) -> bool:
        return self._hotkey_currently_active

    async def start_listening(self):
        """
        Запускает прослушивание клавиатуры.
        """
        if self._hook_installed:
            logger.warning("Listener hook already installed.")
            return

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.error("Cannot start listening: asyncio event loop not found.")
            return

        logger.info(
            "Starting keyboard listener (hotkey: %s | keys=%d | mouse=%s)...",
            self._hotkey_to_string(),
            len(self._hotkey_required),
            'yes' if 'mouse_middle' in self._hotkey_required else 'no'
        )
        keyboard.hook(self._key_event_handler)  # глобальный хук клавиатуры
        # Если требуется мышь
        self._mouse_hook_active = False
        if 'mouse_middle' in self._hotkey_required:
            if not _MOUSE_AVAILABLE:
                raise ImportError("Для использования 'mouse_middle' установите пакет 'mouse': pip install mouse")
            mouse.hook(self._mouse_event_handler)  # type: ignore
            self._mouse_hook_active = True
            logger.info("Mouse hook installed (middle button monitoring)")
        self._hook_installed = True
        logger.info("Keyboard listener started and hook(s) installed.")

    async def stop_listening(self):
        """
        Останавливает прослушивание клавиатуры.
        """
        if not self._hook_installed:
            logger.warning("Listener hook was not installed or already unhooked.")
            return
            
        logger.info("Stopping keyboard listener...")
        keyboard.unhook_all() # Удаляет все хуки клавиатуры
        if getattr(self, '_mouse_hook_active', False) and _MOUSE_AVAILABLE:
            try:
                mouse.unhook(self._mouse_event_handler)  # type: ignore
            except Exception:
                pass
        self._hook_installed = False
        # Сбрасываем состояния
        self._pressed.clear()
        self._hotkey_currently_active = False
        logger.info("Keyboard listener stopped and hook uninstalled (last hotkey: %s).", self._hotkey_to_string())

# Пример использования (для тестирования этого модуля отдельно)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    async def on_start_recording():
        logger.info("== EVENT: START RECORDING ==")

    async def on_stop_recording():
        logger.info("== EVENT: STOP RECORDING ==")

    async def main():
        logger.info("Starting async keyboard listener test.")
        # Пример: задаём хоткей Ctrl + Space (можно любое множество)
        listener = AsyncKeyboardListener(
            start_recording_callback=on_start_recording,
            stop_recording_callback=on_stop_recording,
            hotkey=["ctrl", "space"],
        )
        await listener.start_listening()

        logger.info(
            "Listening for hotkey: %s. Press Esc to exit test.",
            "+".join(listener.current_hotkey),
        )

        # Для выхода из теста по Esc
        exit_event = asyncio.Event()

        def _exit_trigger():
            # Может быть вызван до старта (маловероятно, но на всякий случай)
            if listener._loop is not None:
                listener._loop.call_soon_threadsafe(exit_event.set)
            else:
                # fallback – устанавливаем напрямую (если вдруг вызывают из основного потока)
                exit_event.set()

        keyboard.add_hotkey('esc', _exit_trigger)
        
        try:
            await exit_event.wait() # Ждем события выхода
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, stopping...")
        finally:
            logger.info("Stopping listener in main...")
            # keyboard.unhook_all_hotkeys() # Если использовали add_hotkey для Esc
            await listener.stop_listening()
            logger.info("Test finished.")

    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            logger.info("Asyncio loop closed, exiting.")
        else:
            raise
    except Exception as e:
        logger.error(f"An unexpected error occurred in test: {e}", exc_info=True)
