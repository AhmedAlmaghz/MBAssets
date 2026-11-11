"""Standalone mini-lab for chapter 26: الفصل 26: مشروع تطبيقي متكامل — تدريب Mini-LLM من الصفر."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_bigram() -> None:
    text = 'to be or not to be that is the question to be'
    pairs = Counter(zip(text.split(), text.split()[1:]))
    next_words = {word: sorted(((count, nxt) for (prev, nxt), count in pairs.items() if prev == word), reverse=True) for word in set(text.split())}
    print("learned next-word counts for 'to':", next_words['to'])
    print('most likely continuation:', next_words['to'][0][1])
    print('model parameters:', len(pairs), 'observed bigrams')

if __name__ == "__main__":
    print("Chapter 26 lab: Train a tiny next-character language model")
    run_bigram()
