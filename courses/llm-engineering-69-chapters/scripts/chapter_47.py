"""Standalone mini-lab for chapter 47: الفصل 47: التقييم بواسطة النماذج اللغوية (LLM-as-a-Judge)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_judge() -> None:
    judge_a, judge_b = ([1, 1, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1])
    observed = sum((a == b for a, b in zip(judge_a, judge_b))) / len(judge_a)
    p_a = sum(judge_a) / len(judge_a)
    p_b = sum(judge_b) / len(judge_b)
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    kappa = (observed - expected) / (1 - expected)
    print('raw agreement:', round(observed, 3))
    print("Cohen's kappa:", round(kappa, 3))
    print('resolve disagreement with a blinded human audit.')

if __name__ == "__main__":
    print("Chapter 47 lab: Measure agreement between independent model judges")
    run_judge()
