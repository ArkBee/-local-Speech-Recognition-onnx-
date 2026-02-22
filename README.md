# Local Speech Recognition (GigaAM v3 ONNX + Groq)

Локальное распознавание речи на **GigaAM v3** (ONNX Runtime) с GUI, интеграцией **Groq API** для пунктуации и перевода, и вводом текста в любое окно через буфер обмена.

## Возможности

- **Локальное распознавание** — GigaAM v3 через ONNX Runtime (DirectML / CUDA / CPU)
- **Облачное распознавание** — Groq Whisper Large V3
- **Пунктуация** — Groq Llama 4 Scout добавляет знаки препинания к распознанному тексту
- **Перевод** — Groq Llama 4 Scout переводит на любой язык
- **Горячие клавиши** — 3 режима: сырой текст / +пунктуация / +перевод (hold-to-record)
- **Безопасный ввод** — вставка через буфер обмена + Ctrl+V (без посимвольного набора)
- **4 модели** — v3 RNNT, v3 CTC, v3 E2E RNNT, v3 E2E CTC (переключение в GUI)
- **Автозагрузка моделей** — скачивание ONNX с HuggingFace при первом запуске
- **Системный трей** — сворачивание в трей, автозапуск с Windows

## Структура

```
├── main.py                  # Точка входа
├── settings.json            # Настройки (создаётся при первом запуске)
├── core/
│   ├── audio_recorder.py    # Запись аудио (sounddevice)
│   ├── onnx_recognizer.py   # GigaAM v3 ONNX инференс
│   ├── groq_client.py       # Groq API (Whisper STT + Llama 4)
│   ├── hotkey_manager.py    # Глобальные горячие клавиши
│   ├── text_utils.py        # Замены слов → цифры/символы
│   └── model_downloader.py  # Скачивание моделей с HuggingFace
├── gui/
│   ├── app.py               # Главное окно (tkinter, тёмная тема)
│   ├── test_tab.py          # Тестовая панель распознавания
│   ├── hotkeys_tab.py       # Настройка горячих клавиш
│   └── groq_tab.py          # Настройки Groq (ключи, промпты)
└── models/onnx/             # ONNX модели (скачиваются автоматически)
```

## Установка

```bash
git clone https://github.com/ArkBee/-local-Speech-Recognition-onnx-.git
cd -local-Speech-Recognition-onnx-
```

**Быстрая установка (Windows):**
```bash
setup.bat
```

**Ручная установка:**
```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Запуск

```bash
run.bat
```

Или вручную:
```bash
.venv\Scripts\python main.py
```

При первом запуске модели скачаются автоматически с HuggingFace (~500 МБ для RNNT).

## Настройка

Скопируйте `settings.example.json` в `settings.json` и укажите свои Groq API ключи:

```json
{
    "groq": {
        "keys": ["gsk_YOUR_KEY_HERE"]
    }
}
```

Получить ключ: [console.groq.com](https://console.groq.com)

## Горячие клавиши (по умолчанию)

| Комбинация | Режим |
|---|---|
| `Ctrl + Alt` | Распознавание (сырой текст) |
| `Ctrl + Shift` | Распознавание + пунктуация (Groq) |
| `Ctrl + Shift + Alt` | Распознавание + перевод (Groq) |

Удерживайте комбинацию для записи, отпустите для обработки. Все клавиши настраиваются в GUI.

## GPU ускорение

Поддерживается автоматически через приоритет провайдеров:
1. **CUDA** (NVIDIA + CUDA Toolkit)
2. **DirectML** (любая видеокарта с DirectX 12)
3. **CPU** (fallback)

Для DirectML достаточно установить `onnxruntime-directml` (уже в requirements.txt).

## Зависимости

- Python 3.10+
- onnxruntime-directml
- sounddevice, numpy, sentencepiece
- keyboard (глобальные хоткеи)
- openai (Groq API), huggingface_hub
- pystray, Pillow (системный трей)
