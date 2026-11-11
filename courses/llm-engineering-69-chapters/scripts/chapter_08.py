"""Standalone mini-lab for chapter 08: الفصل الثامن: من الكلمة إلى الجملة — RNN وLSTM وGRU."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_recurrent() -> None:
    inputs = [0.2, 0.8, -0.1, 0.5]
    state, gate = (0.0, 0.65)
    states = []
    for value in inputs:
        candidate = math.tanh(0.7 * state + 0.8 * value)
        state = gate * state + (1.0 - gate) * candidate
        states.append(round(state, 4))
    print('input sequence:', inputs)
    print('gated hidden states:', states)

if __name__ == "__main__":
    print("Chapter 08 lab: A recurrent state with a learned gate")
    run_recurrent()
