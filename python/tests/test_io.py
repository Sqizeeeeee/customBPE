import json

import pytest
from bpe.io import load_tokenizer, save_tokenizer


def make_tok():
    """Мини-токенизатор: 256 байтов + 'th' (116,104) + 'the' (256,101)."""
    vocab = [bytes([i]) for i in range(256)] + [b'th', b'the']
    return dict(
        pattern='x',
        vocab_size=258,
        merges=[(116, 104), (256, 101)],
        vocab=vocab,
        special_tokens={'<|endoftext|>': 258},
    )


def test_roundtrip(tmp_path):
    path = str(tmp_path / 'tok.json')
    save_tokenizer(path, **make_tok())
    result = load_tokenizer(path)
    expected = make_tok()
    assert result['pattern'] == expected['pattern']
    assert result['vocab_size'] == expected['vocab_size']
    assert result['merges'] == expected['merges']
    assert result['vocab'] == expected['vocab']
    assert result['special_tokens'] == expected['special_tokens']


def test_merge_refers_to_future_id(tmp_path):
    tok = make_tok()
    tok['merges'] = [(116, 104), (257, 101)]  # 257 ещё не существует на шаге k=1
    path = str(tmp_path / 'tok.json')
    save_tokenizer(path, **tok)
    with pytest.raises(ValueError):
        load_tokenizer(path)


def test_vocab_bytes_mismatch(tmp_path):
    tok = make_tok()
    tok['vocab'][256] = b'xx'  # должно быть b'th'
    path = str(tmp_path / 'tok.json')
    save_tokenizer(path, **tok)
    with pytest.raises(ValueError):
        load_tokenizer(path)


def test_wrong_vocab_length(tmp_path):
    tok = make_tok()
    tok['vocab'] = tok['vocab'][:-1]
    path = str(tmp_path / 'tok.json')
    save_tokenizer(path, **tok)
    with pytest.raises(ValueError):
        load_tokenizer(path)


def test_bad_version(tmp_path):
    path = tmp_path / 'tok.json'
    save_tokenizer(str(path), **make_tok())
    data = json.loads(path.read_text(encoding='utf-8'))
    data['version'] = 2
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError):
        load_tokenizer(str(path))