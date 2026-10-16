
import re
from typing import List, Tuple

import regex


def split_special(text: str, special_tokens: dict) -> List[Tuple[str, bool]]:

    if not special_tokens:
        if not text:
            return ([])
        return [(text, False)]

    tokens = sorted(special_tokens.keys(), key=lambda f: len(f), reverse=True)
    pattern = '(' + '|'.join(re.escape(t) for t in tokens) + ')'

    parts = re.split(pattern, text)

    res = []
    for i, part in enumerate(parts):
        if len(part) == 0:
            continue

        res.append((part, i % 2 == 1))

    return res


def pretokenize(text: str, pattern: regex.Pattern) -> List[bytes]:

    words = pattern.findall(text)

    res = [word.encode('utf-8') for word in words]

    return res
