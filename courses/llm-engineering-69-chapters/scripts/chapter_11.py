"""Standalone mini-lab for chapter 11: الفصل الحادي عشر: بنية Transformer الكاملة."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_transformer() -> None:
    hidden = [1.0, -1.0, 0.5, 0.0]
    mean = sum(hidden) / len(hidden)
    variance = sum(((x - mean) ** 2 for x in hidden)) / len(hidden)
    normalized = [(x - mean) / math.sqrt(variance + 1e-05) for x in hidden]
    feed_forward = [max(0.0, x * 0.7 + 0.1) for x in normalized]
    output = [x + y for x, y in zip(hidden, feed_forward)]
    print('normalized:', [round(x, 3) for x in normalized])
    print('residual output:', [round(x, 3) for x in output])

if __name__ == "__main__":
    print("Chapter 11 lab: Residual, normalization, and feed-forward blocks")
    run_transformer()
