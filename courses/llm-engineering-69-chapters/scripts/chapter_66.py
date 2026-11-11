"""Standalone mini-lab for chapter 66: الفصل 66: السياق الطويل جدًا (Very Long Context)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_long_context() -> None:
    document = ' '.join((f'section{i} filler' for i in range(40))) + ' attention caches keys and values'
    size, overlap = (12, 3)
    words = document.split()
    chunks = [words[start:start + size] for start in range(0, len(words), size - overlap)]
    query = set('attention keys values'.split())
    ranked = sorted(((len(query & set(chunk)), index, chunk) for index, chunk in enumerate(chunks)), reverse=True)
    print('tokens / chunks:', len(words), len(chunks))
    print('best chunk index / overlap:', ranked[0][1], overlap)
    print('retrieved evidence:', ' '.join(ranked[0][2]))

if __name__ == "__main__":
    print("Chapter 66 lab: Retrieve relevant passages from long-context chunks")
    run_long_context()
