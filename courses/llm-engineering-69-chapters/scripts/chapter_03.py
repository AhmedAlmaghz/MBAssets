"""Standalone mini-lab for chapter 03: الفصل الثالث: Backpropagation والتحسين (Optimization)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_gradient() -> None:
    parameter, rate = (8.0, 0.15)
    trace = []
    for _ in range(30):
        loss = (parameter - 2.0) ** 2
        analytic = 2.0 * (parameter - 2.0)
        epsilon = 1e-05
        numeric = ((parameter + epsilon - 2) ** 2 - (parameter - epsilon - 2) ** 2) / (2 * epsilon)
        trace.append(loss)
        parameter -= rate * analytic
    print('final parameter:', round(parameter, 5))
    print('initial/final loss:', round(trace[0], 4), round(trace[-1], 8))
    print('finite-difference gradient agrees:', abs(analytic - numeric) < 1e-05)

if __name__ == "__main__":
    print("Chapter 03 lab: Gradient descent and finite differences")
    run_gradient()
