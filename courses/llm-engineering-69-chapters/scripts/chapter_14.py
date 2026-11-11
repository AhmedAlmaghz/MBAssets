"""Standalone mini-lab for chapter 14: الفصل الرابع عشر: تحليل التعقيد الحسابي والذاكرة — فهم Transformer بلغة FLOPs والبايتات."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_complexity() -> None:
    batch, length, hidden, layers = (2, 1024, 2048, 24)
    attention = 4 * batch * length ** 2 * hidden
    projections = 8 * batch * length * hidden ** 2
    parameters = 12 * layers * hidden ** 2
    print('attention FLOPs (approx):', f'{attention:,}')
    print('projection/MLP FLOPs (approx):', f'{projections * layers:,}')
    print('parameter count proxy:', f'{parameters:,}')
    print('FP16 weight memory (GiB):', round(parameters * 2 / 2 ** 30, 2))

if __name__ == "__main__":
    print("Chapter 14 lab: Transformer FLOPs and activation memory")
    run_complexity()
