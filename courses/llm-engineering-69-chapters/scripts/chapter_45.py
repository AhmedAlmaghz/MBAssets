"""Standalone mini-lab for chapter 45: الفصل 45: أسس قياس أداء النماذج اللغوية (Foundations of LLM Evaluation) — نسخة مُعاد صياغتها."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_metrics() -> None:
    actual, predicted = ([1, 1, 0, 1, 0, 0], [1, 0, 0, 1, 1, 0])
    tp = sum((a == p == 1 for a, p in zip(actual, predicted)))
    fp = sum((a == 0 and p == 1 for a, p in zip(actual, predicted)))
    fn = sum((a == 1 and p == 0 for a, p in zip(actual, predicted)))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print('precision / recall / F1:', round(precision, 3), round(recall, 3), round(2 * precision * recall / (precision + recall), 3))
    print('accuracy:', round(sum((a == p for a, p in zip(actual, predicted))) / len(actual), 3))

if __name__ == "__main__":
    print("Chapter 45 lab: Calculate classification metrics and uncertainty")
    run_metrics()
