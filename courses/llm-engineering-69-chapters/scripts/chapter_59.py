"""Standalone mini-lab for chapter 59: الفصل 59: تحسين الاستدلال — Quantization وSpeculative Decoding وContinuous Batching."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_inference() -> None:
    weights, scales, quantized = ([0.12, -0.77, 0.34, 1.0], 127, [])
    for value in weights:
        quantized.append(round(value * scales))
    restored = [value / scales for value in quantized]
    acceptance = 0.7
    draft_tokens = 8
    expected_accepted = draft_tokens * acceptance
    print('FP weights:', weights)
    print('INT8 reconstruction:', [round(x, 3) for x in restored])
    print('estimated accepted draft tokens:', round(expected_accepted, 2), 'of', draft_tokens)

if __name__ == "__main__":
    print("Chapter 59 lab: Compare quantization and speculative decoding costs")
    run_inference()
