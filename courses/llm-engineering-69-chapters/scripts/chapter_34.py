"""Standalone mini-lab for chapter 34: الفصل 34: الضبط الفعّال للمعاملات — PEFT وLoRA وQLoRA."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_lora() -> None:
    base = [[0.2, 0.1], [-0.3, 0.4]]
    a, b, alpha, rank = ([[1.0], [0.5]], [[0.2, -0.1]], 2.0, 1)
    delta = [[alpha / rank * sum((b[k][j] * a[i][k] for k in range(rank))) for j in range(2)] for i in range(2)]
    adapted = [[base[i][j] + delta[i][j] for j in range(2)] for i in range(2)]
    print('low-rank update:', delta)
    print('adapted weights:', adapted)
    print('trainable adapter parameters:', len(a) * rank + rank * len(b[0]))

if __name__ == "__main__":
    print("Chapter 34 lab: Apply a low-rank LoRA weight update")
    run_lora()
