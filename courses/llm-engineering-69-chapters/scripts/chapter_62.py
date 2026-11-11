"""Standalone mini-lab for chapter 62: الفصل 62: كشف انحراف البيانات والمفهوم — الحارس الصامت لجودة النموذج في الإنتاج."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_drift() -> None:
    baseline, current = ([0.5, 0.3, 0.2], [0.2, 0.3, 0.5])
    epsilon = 1e-06
    psi = sum(((actual - expected) * math.log((actual + epsilon) / (expected + epsilon)) for expected, actual in zip(baseline, current)))
    print('baseline distribution:', baseline)
    print('current distribution:', current)
    print('population stability index:', round(psi, 4))
    print('alert threshold must be calibrated against historical data.')

if __name__ == "__main__":
    print("Chapter 62 lab: Detect a change in token-frequency distributions")
    run_drift()
