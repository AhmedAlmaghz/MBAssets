"""Standalone mini-lab for chapter 30: الفصل 30: التقليم (Pruning)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_pruning() -> None:
    weights = [0.02, -0.8, 0.12, 0.0, 0.61, -0.03, 0.3, 0.01]
    threshold = sorted((abs(x) for x in weights))[3]
    pruned = [0.0 if abs(x) <= threshold else x for x in weights]
    zeros = sum((x == 0 for x in pruned))
    print('magnitude threshold:', round(threshold, 3))
    print('sparsity:', f'{zeros}/{len(weights)} ({zeros / len(weights):.0%})')
    print('dense storage unchanged without a sparse kernel:', len(weights), 'slots')

if __name__ == "__main__":
    print("Chapter 30 lab: Magnitude pruning and sparsity accounting")
    run_pruning()
