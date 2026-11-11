"""Standalone mini-lab for chapter 38: الفصل 38: استراتيجيات التوليد (Decoding Strategies)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]

def run_decoding() -> None:
    logits = [2.4, 1.7, 0.8, 0.2, -0.3]
    probabilities = softmax(logits)
    ranked = sorted(range(len(logits)), key=logits.__getitem__, reverse=True)
    top_k = ranked[:3]
    cumulative, top_p = (0.0, [])
    for index in ranked:
        top_p.append(index)
        cumulative += probabilities[index]
        if cumulative >= 0.85:
            break
    print('greedy token:', ranked[0])
    print('top-k candidates:', top_k)
    print('top-p candidates:', top_p, '| mass:', round(sum((probabilities[i] for i in top_p)), 3))
    print('temperature=0.5 top token:', max(range(len(logits)), key=lambda i: logits[i] / 0.5))

if __name__ == "__main__":
    print("Chapter 38 lab: Compare greedy, temperature, top-k, and top-p decoding")
    run_decoding()
