"""Standalone mini-lab for chapter 15: الفصل الخامس عشر: تحسينات الانتباه الحديثة — RoPE، ALiBi، MQA/GQA، وSliding Window Attention."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_modern_attention() -> None:
    vector = [1.0, 0.0, -1.0, 0.5]
    angle = 0.25
    rotated = [vector[0] * math.cos(angle) - vector[1] * math.sin(angle), vector[0] * math.sin(angle) + vector[1] * math.cos(angle), vector[2] * math.cos(angle) - vector[3] * math.sin(angle), vector[2] * math.sin(angle) + vector[3] * math.cos(angle)]
    length, window, kv_heads, query_heads = (2048, 256, 4, 16)
    print('RoPE-rotated vector:', [round(x, 3) for x in rotated])
    print('sliding-window visible tokens:', min(length, window))
    print('MQA/GQA KV heads:', kv_heads, 'for', query_heads, 'query heads')

if __name__ == "__main__":
    print("Chapter 15 lab: Rotary positions and sliding-window attention")
    run_modern_attention()
