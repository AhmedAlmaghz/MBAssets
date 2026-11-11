"""Standalone mini-lab for chapter 28: الفصل 28: دمج النماذج (Model Merging)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_merging() -> None:
    base, task_a, task_b = ([0.2, -0.1, 0.5], [0.4, -0.1, 0.5], [0.2, 0.1, 0.7])
    delta_a = [a - b for a, b in zip(task_a, base)]
    delta_b = [a - b for a, b in zip(task_b, base)]
    merged = [b + 0.5 * (a + c) for b, a, c in zip(base, delta_a, delta_b)]
    print('task vectors:', delta_a, delta_b)
    print('merged weights:', [round(x, 3) for x in merged])
    print('all shapes compatible:', len(base) == len(task_a) == len(task_b))

if __name__ == "__main__":
    print("Chapter 28 lab: Merge compatible model weight vectors")
    run_merging()
