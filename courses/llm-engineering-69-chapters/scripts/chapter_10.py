"""Standalone mini-lab for chapter 10: الفصل العاشر: آلية الانتباه الذاتي — قلب الثورة النابض."""

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

def run_attention() -> None:
    queries = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    keys, values = ([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    outputs = []
    for position, query in enumerate(queries):
        scores = [sum((a * b for a, b in zip(query, key))) / math.sqrt(len(query)) if index <= position else -math.inf for index, key in enumerate(keys)]
        weights = softmax(scores)
        outputs.append([sum((weight * value[axis] for weight, value in zip(weights, values))) for axis in range(2)])
        if position == 1:
            assert weights[2] == 0.0
    print('second-token causal weights:', [round(x, 3) for x in softmax([sum((a * b for a, b in zip(queries[1], key))) / math.sqrt(2) if index <= 1 else -math.inf for index, key in enumerate(keys)])])
    print('attention output shape:', (len(outputs), len(outputs[0])))
    print('future positions masked:', True)

if __name__ == "__main__":
    print("Chapter 10 lab: Scaled dot-product causal attention")
    run_attention()
