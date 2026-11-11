"""Standalone mini-lab for chapter 12: الفصل الثاني عشر: التنويعات المعمارية — Encoder-Only، Decoder-Only، Encoder-Decoder."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_masks() -> None:
    length = 5
    causal = [[int(key <= query) for key in range(length)] for query in range(length)]
    padding = [1, 1, 1, 0, 0]
    print('causal mask:')
    for row in causal:
        print(' ', row)
    print('padding-valid positions:', padding)
    print('encoder-decoder mask shape:', (length, 3))

if __name__ == "__main__":
    print("Chapter 12 lab: Encoder, decoder, and cross-attention masks")
    run_masks()
