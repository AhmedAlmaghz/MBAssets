"""Standalone mini-lab for chapter 18: الفصل الثامن عشر: تقنيات معمارية حديثة — RMSNorm، SwiGLU، Multi-Token Prediction، وYaRN/NTK-aware Scaling."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_modern_blocks() -> None:
    values = [1.0, -2.0, 0.5, 3.0]
    rms = math.sqrt(sum((x * x for x in values)) / len(values) + 1e-06)
    normalized = [x / rms for x in values]
    silu = lambda x: x / (1 + math.exp(-x))
    swiglu = [silu(x) * math.tanh(0.8 * x) for x in normalized]
    print('RMSNorm:', [round(x, 3) for x in normalized])
    print('SwiGLU values:', [round(x, 3) for x in swiglu])

if __name__ == "__main__":
    print("Chapter 18 lab: RMSNorm and SwiGLU")
    run_modern_blocks()
