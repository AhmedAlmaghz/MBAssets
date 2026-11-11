"""Standalone mini-lab for chapter 21: الفصل الحادي والعشرون: تصميم Tokenizer مخصص على نطاق واسع."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_bpe() -> None:
    words = ['low', 'lowest', 'newer', 'wider']
    symbols = [list(word) + ['</w>'] for word in words]
    pairs = Counter((pair for word in symbols for pair in zip(word, word[1:])))
    pair, count = pairs.most_common(1)[0]
    merged = ''.join(pair)
    updated = [[merged if tuple(word[i:i + 2]) == pair else word[i] for i in range(len(word)) if i + 1 >= len(word) or tuple(word[i:i + 2]) != pair] for word in symbols]
    print('most frequent pair:', pair, 'count:', count)
    print('merged symbol:', merged)
    print('vocabulary size before/after:', sum(map(len, symbols)), sum(map(len, updated)))

if __name__ == "__main__":
    print("Chapter 21 lab: Learn byte-pair merges from a tiny corpus")
    run_bpe()
