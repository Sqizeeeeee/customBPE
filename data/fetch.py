"""Скачивает текст Википедии нужного размера в один txt-файл.

Установка:  pip install datasets
Запуск:     python fetch.py ru 100 corpus/ru_100mb.txt
            python fetch.py en 100 corpus/en_100mb.txt
            python fetch.py ru 1   corpus/small_1mb.txt
"""
import sys
from pathlib import Path

from datasets import load_dataset

lang = sys.argv[1]            # ru или en
size_mb = int(sys.argv[2])    # целевой размер в МБ
out_path = Path(sys.argv[3])  # куда сохранить

out_path.parent.mkdir(parents=True, exist_ok=True)
limit = size_mb * 1024 * 1024

ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)

written = 0
with open(out_path, "w", encoding="utf-8") as f:
    for row in ds:
        text = row["text"].strip() + "\n\n"
        f.write(text)
        written += len(text.encode("utf-8"))
        if written >= limit:
            break

print(f"Готово: {out_path}, {written / 1024 / 1024:.1f} МБ")
