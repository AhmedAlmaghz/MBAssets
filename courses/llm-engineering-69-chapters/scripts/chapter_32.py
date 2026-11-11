"""Standalone mini-lab for chapter 32: الفصل 32: مصفوفة القرار الاستراتيجي — متى تختار ماذا؟."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_decision() -> None:
    candidates = {'small-local': {'quality': 0.72, 'latency': 0.08, 'cost': 0.01}, 'large-api': {'quality': 0.91, 'latency': 0.8, 'cost': 0.2}}
    weights = {'quality': 0.6, 'latency': 0.25, 'cost': 0.15}
    for name, metrics in candidates.items():
        score = weights['quality'] * metrics['quality'] - weights['latency'] * metrics['latency'] - weights['cost'] * metrics['cost']
        print(name, 'decision score:', round(score, 3))

if __name__ == "__main__":
    print("Chapter 32 lab: Score model choices against project constraints")
    run_decision()
