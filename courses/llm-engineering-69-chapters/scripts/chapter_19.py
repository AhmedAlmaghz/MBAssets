"""Standalone mini-lab for chapter 19: الفصل التاسع عشر: دراسة حالة تطبيقية — تشريح معماري لـ LLaMA 3، Mistral، وDeepSeek-V3."""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict


def run_architecture() -> None:
    models = [{'name': 'encoder', 'bidirectional': True, 'generation': False}, {'name': 'decoder', 'bidirectional': False, 'generation': True}, {'name': 'encoder-decoder', 'bidirectional': True, 'generation': True}]
    for model in models:
        print(model['name'], 'context:', 'both' if model['bidirectional'] else 'causal', '| generation:', model['generation'])

if __name__ == "__main__":
    print("Chapter 19 lab: Compare model architecture configurations")
    run_architecture()
