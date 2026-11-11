"""Standalone mini-lab for chapter 27: الفصل 27: Transfer Learning & Continued Pretraining."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_transfer() -> None:
    general_loss, domain_loss, adapted_domain_loss = (0.42, 0.75, 0.39)
    retained_general_loss = 0.46
    print('domain loss before/after:', domain_loss, '->', adapted_domain_loss)
    print('general loss before/after:', general_loss, '->', retained_general_loss)
    print('domain gain:', round(domain_loss - adapted_domain_loss, 3))
    print('general regression:', round(retained_general_loss - general_loss, 3))

if __name__ == "__main__":
    print("Chapter 27 lab: Measure domain adaptation without forgetting")
    run_transfer()
