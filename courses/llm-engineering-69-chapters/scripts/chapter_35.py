"""Standalone mini-lab for chapter 35: الفصل 35: التعلّم المعزَّز من التغذية الراجعة البشرية (RLHF)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_rlhf() -> None:
    rewards, kl_penalty = ({'helpful': 1.4, 'unsafe': -1.0}, 0.2)
    reference_drift = {'helpful': 0.3, 'unsafe': 0.8}
    objective = {name: reward - kl_penalty * reference_drift[name] for name, reward in rewards.items()}
    print('reward scores:', rewards)
    print('KL-regularized objective:', {key: round(value, 3) for key, value in objective.items()})

if __name__ == "__main__":
    print("Chapter 35 lab: Compare preference rewards and a KL constraint")
    run_rlhf()
