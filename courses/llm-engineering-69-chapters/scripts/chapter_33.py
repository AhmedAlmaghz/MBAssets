"""Standalone mini-lab for chapter 33: الفصل 33: هندسة الأوامر (Prompt Engineering)."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_prompt() -> None:
    prompt = {'role': 'assistant', 'task': 'summarize', 'constraints': ['under 30 words', 'cite supplied text'], 'input': 'Attention combines token representations.'}
    serialized = json.dumps(prompt, ensure_ascii=False, indent=2)
    parsed = json.loads(serialized)
    assert parsed['role'] == 'assistant' and len(parsed['constraints']) == 2
    print(serialized)
    print('required fields validated:', sorted(parsed))

if __name__ == "__main__":
    print("Chapter 33 lab: Build and validate a structured prompt")
    run_prompt()
