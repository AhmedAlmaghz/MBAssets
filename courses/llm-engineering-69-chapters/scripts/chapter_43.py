"""Standalone mini-lab for chapter 43: الفصل 43: نماذج الانتشار (Diffusion Models) — التوليد من الضوضاء."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_diffusion() -> None:
    clean, noise, alpha = ([0.2, -0.4, 0.8], [0.5, 0.3, -0.2], 0.75)
    noisy = [math.sqrt(alpha) * x + math.sqrt(1 - alpha) * n for x, n in zip(clean, noise)]
    estimated_noise = noise
    restored = [(x - math.sqrt(1 - alpha) * e) / math.sqrt(alpha) for x, e in zip(noisy, estimated_noise)]
    print('clean:', clean)
    print('noisy sample:', [round(x, 3) for x in noisy])
    print('reconstructed:', [round(x, 3) for x in restored])

if __name__ == "__main__":
    print("Chapter 43 lab: Add and reverse noise in a toy diffusion process")
    run_diffusion()
