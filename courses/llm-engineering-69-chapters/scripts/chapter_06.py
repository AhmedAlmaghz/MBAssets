"""Standalone mini-lab for chapter 06: الفصل السادس: Tokenization — كيف نُحوّل اللغة إلى أرقام؟."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_tokenizer() -> None:
    text = 'small models learn useful language patterns'
    vocabulary = {word: index for index, word in enumerate(sorted(set(text.split())))}
    encoded = [vocabulary[word] for word in text.split()]
    decoded = ' '.join((next((word for word, idx in vocabulary.items() if idx == token)) for token in encoded))
    print('vocabulary:', vocabulary)
    print('token ids:', encoded)
    print('round trip:', decoded == text)

if __name__ == "__main__":
    print("Chapter 06 lab: A reversible word tokenizer")
    run_tokenizer()
