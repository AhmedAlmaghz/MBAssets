"""Standalone mini-lab for chapter 49: الفصل 49: تقييم السلامة والتحيّز (Safety & Bias Evaluation)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_fairness() -> None:
    groups = {'A': ([1, 1, 0, 0], [1, 0, 0, 0]), 'B': ([1, 1, 0, 0], [1, 1, 1, 0])}
    rates = {}
    for name, (actual, predicted) in groups.items():
        positives = sum((y == 1 for y in actual))
        rates[name] = sum((y == p == 1 for y, p in zip(actual, predicted))) / positives
    print('group true-positive rates:', {key: round(value, 3) for key, value in rates.items()})
    print('absolute disparity:', round(max(rates.values()) - min(rates.values()), 3))
    print('small samples are diagnostic, not a fairness certification.')

if __name__ == "__main__":
    print("Chapter 49 lab: Audit group-level error rates and disparities")
    run_fairness()
