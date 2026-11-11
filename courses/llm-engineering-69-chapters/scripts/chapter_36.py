"""Standalone mini-lab for chapter 36: الفصل 36: التحسين المباشر للتفضيلات (DPO)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_dpo() -> None:
    chosen_margin, rejected_margin, beta = (1.2, -0.3, 0.1)
    logit = beta * (chosen_margin - rejected_margin)
    loss = math.log1p(math.exp(-logit))
    print('preference logit:', round(logit, 4))
    print('DPO logistic loss:', round(loss, 4))
    print('larger chosen-vs-rejected margin lowers loss.')

if __name__ == "__main__":
    print("Chapter 36 lab: Compute a stable DPO preference loss")
    run_dpo()
