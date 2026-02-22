import re
from difflib import get_close_matches
from typing import Dict, List

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
    "ноль": "0", "нуль": "0",
    "один": "1", "два": "2", "три": "3", "четыре": "4", "пять": "5",
    "шесть": "6", "семь": "7", "восемь": "8", "девять": "9", "десять": "10",
}

NUMERIC_VARIANTS: Dict[str, str] = {
    **WORD_TO_NUMERIC_MAP,
    "одна": "1", "одно": "1", "одну": "1", "одним": "1", "одного": "1", "одина": "1",
    "две": "2", "дво": "2",
    "трое": "3", "четверо": "4", "пятеро": "5",
    "шестер": "6", "семеро": "7", "восьмеро": "8", "девятеро": "9",
}

WORD_TO_PUNCTUATION_MAP: Dict[str, str] = {
    "запятая": ",", "точка": ".", "двоеточие": ":", "тире": "-", "дефис": "-",
}


def process_transcription(
    text: str,
    *,
    fuzzy_numeric: bool = False,
    collapse_full_repeat: bool = False,
) -> str:
    if not text:
        return ""

    processed = text
    for phrase, replacement in WORD_REPLACEMENTS.items():
        processed = re.compile(re.escape(phrase), re.IGNORECASE).sub(replacement, processed)

    words = processed.split()
    result: List[str] = []
    numeric_keys: List[str] = list(NUMERIC_VARIANTS.keys()) if fuzzy_numeric else []

    for word in words:
        wl = word.lower()

        num = None
        if fuzzy_numeric:
            num = NUMERIC_VARIANTS.get(wl)
            if num is None and len(wl) >= 4:
                close = get_close_matches(wl, numeric_keys, n=1, cutoff=0.85)
                if close:
                    num = NUMERIC_VARIANTS.get(close[0])
        else:
            num = WORD_TO_NUMERIC_MAP.get(wl)

        if num is not None:
            result.append(num)
        elif wl in WORD_TO_PUNCTUATION_MAP:
            if result:
                result[-1] += WORD_TO_PUNCTUATION_MAP[wl]
            else:
                result.append(WORD_TO_PUNCTUATION_MAP[wl])
        else:
            result.append(word)

    if collapse_full_repeat:
        result = _collapse_repeated_sequence(result)

    return " ".join(result)


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
