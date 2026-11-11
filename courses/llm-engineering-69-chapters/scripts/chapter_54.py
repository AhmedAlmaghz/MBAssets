"""Standalone mini-lab for chapter 54: الفصل 54: التلخيص والإجابة على الأسئلة."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_rouge() -> None:
    reference = 'attention combines token values using learned weights'.split()
    summary = 'attention combines values using weights'.split()
    previous = list(range(len(reference) + 1))
    for token in summary:
        current = [previous[0] + 1]
        for index, target in enumerate(reference, 1):
            current.append(previous[index - 1] + 1 if token == target else max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision, recall = (lcs / len(summary), lcs / len(reference))
    f1 = 2 * precision * recall / (precision + recall)
    print('LCS length:', lcs)
    print('ROUGE-L precision/recall/F1:', round(precision, 3), round(recall, 3), round(f1, 3))

if __name__ == "__main__":
    print("Chapter 54 lab: Score a summary with ROUGE-L")
    run_rouge()
