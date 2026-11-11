"""Standalone mini-lab for chapter 65: الفصل 65: نماذج الاستدلال الحديثة — Test-Time Compute Scaling وثورة "التفكير قبل الإجابة"."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_test_time() -> None:
    candidates = ['A', 'B', 'A', 'A', 'C', 'B', 'A']
    counts = Counter(candidates)
    winner, votes = counts.most_common(1)[0]
    print('candidates generated:', len(candidates))
    print('self-consistency vote:', dict(counts))
    print('selected:', winner, f'({votes}/{len(candidates)})')
    print('increase test-time budget only when expected quality gain justifies cost.')

if __name__ == "__main__":
    print("Chapter 65 lab: Allocate a test-time compute budget to voting")
    run_test_time()
