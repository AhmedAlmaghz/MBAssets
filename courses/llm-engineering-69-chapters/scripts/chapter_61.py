"""Standalone mini-lab for chapter 61: الفصل 61: بنية LLMOps — Versioning وCanary Deployment وSemantic Caching."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_llmops() -> None:
    versions = {'stable': {'requests': 1000, 'errors': 12, 'p95_ms': 220}, 'canary': {'requests': 100, 'errors': 1, 'p95_ms': 205}}
    rates = {name: data['errors'] / data['requests'] for name, data in versions.items()}
    allowed_gap = 0.02
    print('error rates:', {key: round(value, 3) for key, value in rates.items()})
    print('canary decision:', 'continue' if rates['canary'] <= rates['stable'] + allowed_gap else 'rollback')
    print('p95 latency:', {key: value['p95_ms'] for key, value in versions.items()})

if __name__ == "__main__":
    print("Chapter 61 lab: Compare canary and stable deployment metrics")
    run_llmops()
