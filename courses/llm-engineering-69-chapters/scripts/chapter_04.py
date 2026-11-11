"""Standalone mini-lab for chapter 04: الفصل الرابع: التقييم والتعميم — Overfitting وRegularization وPerplexity."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_validation() -> None:
    train = [0.8, 0.7, 0.9, 0.6]
    validation = [0.4, 0.5, 0.3]
    tokens = [0.2, 0.3, 0.25]
    perplexity = math.exp(-sum((math.log(p) for p in tokens)) / len(tokens))
    print('train mean loss:', round(statistics.mean(train), 3))
    print('validation mean loss:', round(statistics.mean(validation), 3))
    print('generalization gap:', round(statistics.mean(validation) - statistics.mean(train), 3))
    print('toy-token perplexity:', round(perplexity, 3))

if __name__ == "__main__":
    print("Chapter 04 lab: Generalization and perplexity")
    run_validation()
