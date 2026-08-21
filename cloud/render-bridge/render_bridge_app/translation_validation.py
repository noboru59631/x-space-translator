"""Validate preservation-sensitive values in translated transcript segments."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
TICKER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(BTC|ETH|SOL|XRP|USDT|USDC)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(
    r"[-+]?\s*[$¥€£]?\s*"
    r"(?P<number>\d[\d,]*(?:\.\d+)?)"
    r"\s*(?P<scale>"
    r"(?:thousand|million|billion|trillion)\b|"
    r"[kmbt](?![A-Za-z])|万|億|兆"
    r")?"
    r"(?:\s*(?:%|percent|％))?",
    re.IGNORECASE,
)
WORD_NUMBER_PATTERN = re.compile(
    r"\b(?P<number>zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)"
    r"(?:\s+(?P<scale>thousand|million|billion|trillion))?\b",
    re.IGNORECASE,
)
MONTH_PATTERN = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\b"
)
TRAILING_URL_PUNCTUATION = ".,;:!?)]}、。！？）］｝"
SCALES = {
    "": Decimal(1),
    "k": Decimal(1_000),
    "thousand": Decimal(1_000),
    "m": Decimal(1_000_000),
    "million": Decimal(1_000_000),
    "b": Decimal(1_000_000_000),
    "billion": Decimal(1_000_000_000),
    "t": Decimal(1_000_000_000_000),
    "trillion": Decimal(1_000_000_000_000),
    "万": Decimal(10_000),
    "億": Decimal(100_000_000),
    "兆": Decimal(1_000_000_000_000),
}
SMALL_KANJI_NUMBERS = {
    "0": "零",
    "1": "一",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
    "10": "十",
}
WORD_NUMBERS = {
    "zero": Decimal(0),
    "one": Decimal(1),
    "two": Decimal(2),
    "three": Decimal(3),
    "four": Decimal(4),
    "five": Decimal(5),
    "six": Decimal(6),
    "seven": Decimal(7),
    "eight": Decimal(8),
    "nine": Decimal(9),
    "ten": Decimal(10),
    "first": Decimal(1),
    "second": Decimal(2),
    "third": Decimal(3),
    "fourth": Decimal(4),
    "fifth": Decimal(5),
    "sixth": Decimal(6),
    "seventh": Decimal(7),
    "eighth": Decimal(8),
    "ninth": Decimal(9),
    "tenth": Decimal(10),
}
MONTH_NUMBERS = {
    month: Decimal(index)
    for index, month in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        start=1,
    )
}


@dataclass(frozen=True)
class PreservationCheck:
    number_ok: bool
    url_ok: bool
    ticker_ok: bool

    @property
    def ok(self) -> bool:
        return self.number_ok and self.url_ok and self.ticker_ok


def normalized_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value)


def extract_urls(value: str) -> Counter[str]:
    return Counter(
        match.group(0).rstrip(TRAILING_URL_PUNCTUATION)
        for match in URL_PATTERN.finditer(normalized_text(value))
    )


def extract_tickers(value: str) -> Counter[str]:
    return Counter(
        match.group(1).upper()
        for match in TICKER_PATTERN.finditer(normalized_text(value))
    )


def extract_numbers(value: str) -> Counter[str]:
    numbers: Counter[str] = Counter()
    for match in NUMBER_PATTERN.finditer(normalized_text(value)):
        raw_number = match.group("number").replace(",", "")
        scale_name = (match.group("scale") or "").lower()
        try:
            number = Decimal(raw_number) * SCALES[scale_name]
        except (InvalidOperation, KeyError):
            continue
        normalized = format(number.normalize(), "f")
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")
        numbers[normalized or "0"] += 1
    return numbers


def extract_english_word_numbers(value: str) -> Counter[str]:
    numbers: Counter[str] = Counter()
    for match in WORD_NUMBER_PATTERN.finditer(normalized_text(value)):
        number = WORD_NUMBERS[match.group("number").lower()]
        scale_name = (match.group("scale") or "").lower()
        normalized = format((number * SCALES[scale_name]).normalize(), "f")
        numbers[normalized or "0"] += 1
    for match in MONTH_PATTERN.finditer(normalized_text(value)):
        numbers[str(MONTH_NUMBERS[match.group(1)])] += 1
    return numbers


def numbers_match(original: str, translation: str) -> bool:
    expected = extract_numbers(original)
    actual = extract_numbers(translation)
    if expected == actual:
        return True

    optional_numbers = extract_english_word_numbers(original)
    for number, actual_count in actual.items():
        extra_count = actual_count - expected[number]
        if extra_count > 0:
            expected[number] += min(extra_count, optional_numbers[number])

    normalized_translation = normalized_text(translation)
    supplemented = actual.copy()
    for number, expected_count in expected.items():
        missing = expected_count - supplemented[number]
        kanji = SMALL_KANJI_NUMBERS.get(number)
        if missing > 0 and kanji:
            supplemented[number] += min(missing, normalized_translation.count(kanji))
    if set(expected) != set(supplemented):
        return False
    return all(supplemented[number] <= expected[number] for number in supplemented)


def validate_preservation(original: str, translation: str) -> PreservationCheck:
    return PreservationCheck(
        number_ok=numbers_match(original, translation),
        url_ok=extract_urls(original) == extract_urls(translation),
        ticker_ok=extract_tickers(original) == extract_tickers(translation),
    )
