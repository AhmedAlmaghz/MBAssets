"""Standalone mini-lab for chapter 07: الفصل السابع: Word2Vec — كيف تتعلّم الآلة معنى الكلمات؟."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def cosine(left: list[float], right: list[float]) -> float:
    denominator = math.sqrt(sum((x * x for x in left)) * sum((y * y for y in right)))
    return sum((x * y for x, y in zip(left, right))) / denominator if denominator else 0.0

def run_embeddings() -> None:
    corpus = ['the cat sat', 'the cat slept', 'the dog sat', 'a dog ran']
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sentence in corpus:
        words = sentence.split()
        for index, word in enumerate(words):
            for neighbor in words[max(0, index - 1):index] + words[index + 1:index + 2]:
                counts[word][neighbor] += 1
    target = [counts['cat'].get(w, 0) for w in ('sat', 'slept', 'dog', 'ran')]
    related = [counts['dog'].get(w, 0) for w in ('sat', 'slept', 'dog', 'ran')]
    print('context vector(cat):', target)
    print('context vector(dog):', related)
    print('cosine(cat, dog):', round(cosine(target, related), 3))

if __name__ == "__main__":
    print("Chapter 07 lab: Context co-occurrence and cosine similarity")
    run_embeddings()
