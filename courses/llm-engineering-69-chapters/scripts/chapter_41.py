"""Standalone mini-lab for chapter 41: الفصل 41: نماذج الرؤية واللغة (VLMs) — LLaVA وFlamingo وBLIP-2."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_vlm() -> None:
    visual_width, language_width, tokens = (4, 3, 2)
    projector = [[0.1 * (row + col + 1) for col in range(language_width)] for row in range(visual_width)]
    visual_tokens = [[1.0, 0.5, 0.0, 1.0], [0.0, 1.0, 0.5, 0.5]]
    projected = [[sum((vector[i] * projector[i][j] for i in range(visual_width))) for j in range(language_width)] for vector in visual_tokens]
    print('visual feature shape:', (tokens, visual_width))
    print('projected language-token shape:', (tokens, language_width))
    print('projected tokens:', [[round(x, 2) for x in row] for row in projected])

if __name__ == "__main__":
    print("Chapter 41 lab: Project visual features into language-token space")
    run_vlm()
