"""Standalone mini-lab for chapter 01: الفصل الأول: مقدّمات رياضية سريعة — اللغة السرّية التي يتحدّثها كل نموذج ذكاء اصطناعي."""

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

def cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum((x * x for x in left)) * sum((y * y for y in right)))
    return sum((x * y for x, y in zip(left, right))) / denominator if denominator else 0.0

def run_math() -> None:
    vector_a, vector_b = ([1.0, 2.0, 0.0], [2.0, 1.0, 2.0])
    probabilities = softmax([1.2, 0.3, -0.5])
    entropy = -sum((p * math.log(p) for p in probabilities))
    print('dot product:', sum((a * b for a, b in zip(vector_a, vector_b))))
    print('cosine similarity:', round(cosine(vector_a, vector_b), 4))
    print('softmax:', [round(p, 4) for p in probabilities])
    print('entropy (nats):', round(entropy, 4))

if __name__ == "__main__":
    print("Chapter 01 lab: Linear algebra and probability")
    run_math()
