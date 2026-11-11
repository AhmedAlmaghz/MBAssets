"""Standalone mini-lab for chapter 29: الفصل 29: التقطير المعرفي (Knowledge Distillation)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponents = [math.exp(value - maximum) for value in values]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]

def run_distillation() -> None:
    teacher_logits, student_logits, temperature = ([3.2, 1.1, 0.2], [2.5, 1.4, 0.4], 2.0)
    teacher = softmax([x / temperature for x in teacher_logits])
    student = softmax([x / temperature for x in student_logits])
    kl = sum((p * math.log(p / q) for p, q in zip(teacher, student)))
    print('teacher probabilities:', [round(x, 3) for x in teacher])
    print('student probabilities:', [round(x, 3) for x in student])
    print('soft-target KL divergence:', round(kl, 5))

if __name__ == "__main__":
    print("Chapter 29 lab: Distill a teacher distribution into a student")
    run_distillation()
