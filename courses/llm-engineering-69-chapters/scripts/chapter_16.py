"""Standalone mini-lab for chapter 16: الفصل السادس عشر: Mixture of Experts — فصل الحجم عن التكلفة."""

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

def run_moe() -> None:
    scores = [0.2, 1.4, 0.8, -0.1]
    top_two = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:2]
    weights = softmax([scores[index] for index in top_two])
    expert_load = Counter(top_two)
    print('selected experts:', top_two)
    print('mixture weights:', [round(x, 3) for x in weights])
    print('active parameters:', '2 of', len(scores), 'experts')
    print('routing counts:', dict(expert_load))

if __name__ == "__main__":
    print("Chapter 16 lab: Top-k mixture-of-experts routing")
    run_moe()
