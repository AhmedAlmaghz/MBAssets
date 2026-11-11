"""Standalone mini-lab for chapter 37: الفصل 37: تقنيات محاذاة حديثة — KTO وORPO وConstitutional AI."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_alignment() -> None:
    chosen, rejected = (0.8, 0.2)
    kto_chosen = -math.log(max(chosen, 1e-09))
    kto_rejected = -math.log(max(1 - rejected, 1e-09))
    orpo_odds = math.log(chosen / (1 - chosen)) - math.log(rejected / (1 - rejected))
    print('KTO chosen/rejected losses:', round(kto_chosen, 3), round(kto_rejected, 3))
    print('ORPO log-odds preference margin:', round(orpo_odds, 3))

if __name__ == "__main__":
    print("Chapter 37 lab: Compare simple KTO and ORPO objectives")
    run_alignment()
