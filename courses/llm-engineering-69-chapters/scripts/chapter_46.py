"""Standalone mini-lab for chapter 46: الفصل 46: معايير القياس الكبرى (Major Benchmarks: MMLU, HellaSwag, GSM8K, HumanEval)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_benchmark() -> None:
    examples = [{'id': 'a', 'expected': 'yes', 'output': 'yes'}, {'id': 'b', 'expected': 'no', 'output': 'no'}, {'id': 'c', 'expected': 'yes', 'output': 'no'}]
    results = [{'id': item['id'], 'correct': item['expected'] == item['output']} for item in examples]
    print('per-example results:', results)
    print('accuracy:', round(sum((item['correct'] for item in results)) / len(results), 3))
    fingerprint = hashlib.sha256('|'.join((item['id'] for item in examples)).encode()).hexdigest()[:12]
    print('dataset fingerprint:', fingerprint)

if __name__ == "__main__":
    print("Chapter 46 lab: Run a deterministic miniature benchmark harness")
    run_benchmark()
