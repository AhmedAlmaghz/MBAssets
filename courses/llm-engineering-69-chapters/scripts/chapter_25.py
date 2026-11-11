"""Standalone mini-lab for chapter 25: الفصل 25: من Base إلى نموذج قابل للاستخدام — Supervised Fine-Tuning والبيانات الاصطناعية."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_sft() -> None:
    prompt, answer = ('User: define attention\nAssistant:', ' attention weights combine context')
    tokens = (prompt + answer).split()
    labels = [-100] * len(prompt.split()) + list(range(len(prompt.split()), len(tokens)))
    print('tokens:', tokens)
    print('loss-masked labels:', labels)
    print('supervised completion tokens:', sum((label != -100 for label in labels)))

if __name__ == "__main__":
    print("Chapter 25 lab: Mask prompt tokens in supervised fine-tuning")
    run_sft()
