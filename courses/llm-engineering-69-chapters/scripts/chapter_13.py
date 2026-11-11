"""Standalone mini-lab for chapter 13: الفصل الثالث عشر: نحو الاستدلال السريع — KV Caching وFlash Attention ومقدمة في Quantization."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_kv_cache() -> None:
    prompt_length, new_tokens, layers, heads, dim = (128, 16, 8, 8, 64)
    tokens_recomputed = sum((prompt_length + step for step in range(new_tokens)))
    cache_elements = 2 * layers * heads * (prompt_length + new_tokens) * dim
    bytes_fp16 = cache_elements * 2
    print('tokens processed with cache:', prompt_length + new_tokens)
    print('tokens processed without cache:', tokens_recomputed)
    print('KV cache FP16 (MiB):', round(bytes_fp16 / 2 ** 20, 2))
    print('INT8 cache estimate (MiB):', round(bytes_fp16 / 2 / 2 ** 20, 2))

if __name__ == "__main__":
    print("Chapter 13 lab: KV cache growth and quantized memory")
    run_kv_cache()
