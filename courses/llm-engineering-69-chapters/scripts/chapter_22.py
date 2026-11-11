"""Standalone mini-lab for chapter 22: الفصل الثاني والعشرون: تصميم المعمارية وScaling Laws — فن اختيار الحجم الأمثل لنموذجك."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_scaling() -> None:
    parameters, tokens, coefficient, exponent = (1000000000.0, 20000000000.0, 1.0, 0.34)
    compute = 6 * parameters * tokens
    proxy = coefficient * (parameters ** (-exponent) + tokens ** (-exponent))
    print('training FLOPs proxy:', f'{compute:.3e}')
    print('loss scaling proxy:', f'{proxy:.5f}')
    print('double parameters only compute multiplier:', 2.0)

if __name__ == "__main__":
    print("Chapter 22 lab: Estimate compute with a scaling-law proxy")
    run_scaling()
