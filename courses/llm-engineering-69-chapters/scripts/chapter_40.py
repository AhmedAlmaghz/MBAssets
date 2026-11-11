"""Standalone mini-lab for chapter 40: الفصل 40: CLIP — عندما تتعلّم الصورة والكلمة لغة مشتركة."""

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

def run_clip() -> None:
    images = [[1.0, 0.0], [0.0, 1.0]]
    texts = [[0.9, 0.1], [0.1, 0.9]]
    scores = [[cosine(image, text) / 0.07 for text in texts] for image in images]
    print('image-text similarity logits:')
    for row in scores:
        print(' ', [round(x, 2) for x in row])
    print('matched-pair retrieval:', [max(range(2), key=row.__getitem__) for row in scores])

if __name__ == "__main__":
    print("Chapter 40 lab: Rank images with a shared-space CLIP similarity")
    run_clip()
