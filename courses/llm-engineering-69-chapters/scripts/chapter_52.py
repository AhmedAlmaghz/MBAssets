"""Standalone mini-lab for chapter 52: الفصل 52: الترجمة الآلية (Machine Translation)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_bleu() -> None:
    reference = 'the cat is on the mat'.split()
    candidate = 'the cat sat on mat'.split()
    matches = sum((Counter(zip(candidate, candidate[1:])) & Counter(zip(reference, reference[1:]))).values())
    total = max(len(candidate) - 1, 1)
    brevity = min(1.0, math.exp(1 - len(reference) / len(candidate)))
    score = brevity * matches / total
    print('clipped bigram precision:', round(matches / total, 3))
    print('brevity penalty:', round(brevity, 3))
    print('toy BLEU-2 (without smoothing):', round(score, 3))

if __name__ == "__main__":
    print("Chapter 52 lab: Calculate corpus BLEU with brevity penalty")
    run_bleu()
