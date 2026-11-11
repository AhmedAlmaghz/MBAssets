"""Standalone mini-lab for chapter 17: الفصل السابع عشر: نماذج الفضاء الحالة (State Space Models) وMamba."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_state_space() -> None:
    inputs = [1.0, 0.0, 0.5, -0.2, 0.8]
    state, decay, gain = (0.0, 0.8, 0.25)
    outputs = []
    for value in inputs:
        state = decay * state + gain * value
        outputs.append(round(state, 4))
    print('input:', inputs)
    print('linear recurrent states:', outputs)
    print('work grows linearly with sequence length:', len(inputs))

if __name__ == "__main__":
    print("Chapter 17 lab: A recurrent state-space sequence model")
    run_state_space()
