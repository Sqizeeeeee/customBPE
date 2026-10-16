"""python/bpe/io.py: всё, что связано с файлами.

Чтение корпуса, чтение паттерна, сохранение и загрузка токенизатора
в общем JSON-формате (см. spec/format.md).
"""
import json
from typing import Iterator

import regex


def load_pattern(path: str) -> "regex.Pattern":
    """Читает паттерн из файла (spec/pattern.txt) и компилирует его."""

    with open(path, 'r', encoding='utf-8') as f:

        pattern = f.read().strip()

    return regex.compile(pattern)



def iter_lines(path: str) -> Iterator[str]:
    """Читает текстовый файл потоком, по одной строке (encoding utf-8). """

    with open(path, 'r', encoding='utf-8') as f:

        for line in f:

            yield line


def save_tokenizer(path: str, pattern: str, vocab_size: int, merges: list[tuple[int, int]],
    vocab: list[bytes], special_tokens: dict[str, int]) -> None:
    """Сохраняет токенизатор в JSON (формат из spec/format.md)."""

    data = {'version': 1,
            'pattern': pattern,
            'vocab_size': vocab_size,
            "merges": merges,
            "vocab": [b.hex() for b in vocab],
            "special_tokens": special_tokens
            }

    with open(path, 'w', encoding='utf-8') as f:

        json.dump(data, f, ensure_ascii=False)



def load_tokenizer(path: str) -> dict:
    """Загружает токенизатор из JSON и проверяет инварианты.

    Возвращает словарь с полями: pattern, vocab_size, merges (список кортежей),
    vocab (список bytes), special_tokens.
    """

    with open(path, 'r', encoding='utf-8') as f:

        data = json.load(f)

        if data['version'] != 1:
            raise ValueError('version must be 1')

        vocab_len = len(data['vocab'])
        merges_len = len(data['merges'])

        if vocab_len != 256 + merges_len:
            raise ValueError("len vocab != 256 + len merges")

        vocab_bytes = [bytes.fromhex(h) for h in data['vocab']]

        if not all(vocab_bytes[i] == bytes([i]) for i in range(256)):
            raise ValueError("first 256 tokens must be bytes from 0 to 255")

        merges = [tuple(m) for m in data['merges']]

        for k, (a, b) in enumerate(merges):

            if (not (0 <= a < 256 + k)) or (not (0 <= b < 256 + k)):
                raise ValueError(f"merge {k}: ids {(a, b)} must be in [0, {256 + k})")

            if vocab_bytes[256 + k] != vocab_bytes[a] + vocab_bytes[b]:
                raise ValueError(f"vocab[{256 + k}] != vocab[{a}] + vocab[{b}]")
            
        return {
            'pattern': data['pattern'],
            'vocab_size': data['vocab_size'],
            'merges': merges,
            'vocab': vocab_bytes,
            'special_tokens': data['special_tokens']
        }