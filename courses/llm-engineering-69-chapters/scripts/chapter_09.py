"""Standalone mini-lab for chapter 09: الفصل التاسع: الدافع وراء Self-Attention."""

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
    print("Chapter 09 lab: Alignment attention over encoder states")
    run_alignment()
