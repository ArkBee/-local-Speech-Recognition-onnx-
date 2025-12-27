from difflib import get_close_matches
from typing import Dict, List

# --- Keyword mappings ---
WORD_REPLACEMENTS: Dict[str, str] = {
    "вопросительный знак": "?",
    "восклицательный знак": "!",
    "точка с запятой": ";",
    "открыть скобку": "(",
    "закрыть скобку": ")",
    "открыть квадратную скобку": "[",
    "закрыть квадратную скобку": "]",
    "открыть фигурную скобку": "{",
    "закрыть фигурную скобку": "}",
    "знак процента": "%",
    "доллар": "$",
    "евро": "€",
    "плюс": "+",
    "минус": "-",
    "равно": "=",
    "звезда": "*",
    "слеш": "/",
    "новая строка": "\n",
    "абзац": "\n\n",
}

WORD_TO_NUMERIC_MAP: Dict[str, str] = {
    "ноль": "0",
    "нуль": "0",
    "один": "1",
    "два": "2",
    "три": "3",
    "четыре": "4",
    "пять": "5",
    "шесть": "6",
    "семь": "7",
    "восемь": "8",
    "девять": "9",
    "десять": "10",
}

NUMERIC_VARIANTS: Dict[str, str] = {
    **WORD_TO_NUMERIC_MAP,
    "одна": "1",
    "одно": "1",
    "одну": "1",
    "одним": "1",
    "одного": "1",
    "одина": "1",
    "две": "2",
    "дво": "2",
    "трое": "3",
    "четверо": "4",
    "пятеро": "5",
    "шестер": "6",
    "семеро": "7",
    "восьмеро": "8",
    "девятеро": "9",
}

WORD_TO_PUNCTUATION_MAP: Dict[str, str] = {
    "запятая": ",",
    "точка": ".",
    "двоеточие": ":",
    "тире": "-",
    "дефис": "-",
}


def _collapse_repeated_sequence(words: List[str]) -> List[str]:
    n = len(words)
    if n < 2:
        return words

    for size in range(1, n // 2 + 1):
        if n % size != 0:
            continue
        repeats = n // size
        if repeats < 2:
            continue
        pattern = words[:size]
        if pattern * repeats == words:
            return pattern

    return words


def process_transcription(
    text: str,
    *,
    fuzzy_numeric: bool = False,
    collapse_full_repeat: bool = False,
) -> str:
    if not text:
        return ""

    processed_text = text
    # Case-insensitive replacement for phrases
    import re
    for phrase, replacement in WORD_REPLACEMENTS.items():
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        processed_text = pattern.sub(replacement, processed_text)

    words = processed_text.split()
    result_words: List[str] = []

    numeric_keys: List[str] = list(NUMERIC_VARIANTS.keys()) if fuzzy_numeric else []
    
    for word in words:
        word_lower = word.lower()

        numeric_value = None
        if fuzzy_numeric:
            numeric_value = NUMERIC_VARIANTS.get(word_lower)
            if numeric_value is None:
                close = get_close_matches(word_lower, numeric_keys, n=1, cutoff=0.8)
                if close:
                    numeric_value = NUMERIC_VARIANTS.get(close[0])
        else:
            numeric_value = WORD_TO_NUMERIC_MAP.get(word_lower)

        if numeric_value is not None:
            result_words.append(numeric_value)
        elif word_lower in WORD_TO_PUNCTUATION_MAP:
            if result_words:
                result_words[-1] += WORD_TO_PUNCTUATION_MAP[word_lower]
            else:
                result_words.append(WORD_TO_PUNCTUATION_MAP[word_lower])
        else:
            result_words.append(word)

    if collapse_full_repeat:
        result_words = _collapse_repeated_sequence(result_words)
    return " ".join(result_words)


def extract_transcription_text(transcription_result) -> str:
    if isinstance(transcription_result, str):
        return transcription_result

    if isinstance(transcription_result, list) and transcription_result:
        texts: List[str] = []
        for item in transcription_result:
            if isinstance(item, dict):
                texts.append(item.get("text", ""))
            else:
                texts.append(str(item))
        return " ".join(filter(None, texts))

    if transcription_result:
        return str(transcription_result)

    return ""
