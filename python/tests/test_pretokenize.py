import json
import pathlib

from bpe.io import load_pattern
from bpe.pretokenize import pretokenize, split_special

ROOT = pathlib.Path(__file__).resolve().parents[2]
EOT = '<|endoftext|>'


def test_golden_pretokenize():
    golden = json.loads((ROOT / 'spec/golden/pretokenize.json').read_text(encoding='utf-8'))
    pattern = load_pattern(str(ROOT / golden['pattern_file']))
    for case in golden['cases']:
        expected = [c.encode('utf-8') for c in case['chunks']]
        assert pretokenize(case['text'], pattern) == expected, case['text']


def test_split_no_special_tokens():
    assert split_special('Hello', {}) == [('Hello', False)]
    assert split_special('', {}) == []


def test_split_tokens_given_but_absent_in_text():
    assert split_special('Hello', {EOT: 256}) == [('Hello', False)]
    assert split_special('', {EOT: 256}) == []


def test_split_special_at_both_ends():
    assert split_special(f'{EOT}Hello{EOT}', {EOT: 256}) == [
        (EOT, True), ('Hello', False), (EOT, True),
    ]


def test_split_special_in_middle():
    assert split_special(f'Hello{EOT}world', {EOT: 256}) == [
        ('Hello', False), (EOT, True), ('world', False),
    ]


def test_split_two_special_tokens_in_a_row():
    tokens = {'<|a|>': 256, '<|b|>': 257}
    assert split_special('<|a|><|b|>', tokens) == [('<|a|>', True), ('<|b|>', True)]


def test_split_longest_special_token_wins():
    tokens = {'<|a|>': 256, '<|a|>x': 257}
    assert split_special('<|a|>xy', tokens) == [('<|a|>x', True), ('y', False)]
    